# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Initial expert slot layout and runtime slot-table state for Dynamic EPLB.

When `TPU_MOE_EPLB_DYNAMIC=1` is enabled, each target RoutedExperts layer is
initialized with a blind contiguous-plus-one layout (zero prior profiling):
  * EP rank `r` holds its contiguous block `[r * (E // EP) .. (r + 1) * (E // EP) - 1]`
    plus the first expert of the next rank `((r + 1) % EP) * (E // EP)` in one
    redundant active slot (`DEFAULT_EXTRA_SLOTS = 1`);
  * `DEFAULT_STAGING = 2` spare physical slots sit behind the active slots on
    every rank for background weight staging (`expert_relayout.py`);
  * `expert_slot_table` lists, per logical expert, the physical slot ids
    `r * slots + j` holding it, padded with -1 to `replica_cap = EP`.

Only the fused EP kernel reads slot ids, so a placed layer must be armed for
it; `check_layer_served` refuses anything else.
"""

import re
import threading
from dataclasses import dataclass

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# `model.layers.7.ffn.experts` (a module prefix) and `layers.7.ffn.experts.3.w1`
# (a checkpoint tensor) both name layer 7.
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# A DSpark drafter's MoE layer i is built from the checkpoint's `mtp.i.*`
# tensors; the drafter module itself is named `model.layers.i`.
_DRAFTER_TENSOR_RE = re.compile(r"^mtp\.(\d+)\.")

TARGET = "target"
DRAFT = "draft"

DEFAULT_EXTRA_SLOTS = 1
DEFAULT_STAGING = 2

# The layer attribute the slot table is registered under.
EXPERT_SLOT_ATTR = "_tpu_expert_slot"
# The layer attribute recording which group (target or draft) placed it.
_GROUP_ATTR = "_tpu_expert_group"
# The layer attribute naming its (group, layer index) under the layout.
LAYOUT_KEY_ATTR = "_tpu_layout_key"

# (group, layer index) pairs whose expert map follows the layout in this
# process. The fetch-time filter consults the layout only for these, so a model
# whose MoE method never applied it keeps the default filter.
_APPLIED_LAYERS: set[tuple[str, int]] = set()
# (group, layer index) -> the placed layer module, for the dynamic re-layout
# (its tables to install, its weights to move). Modules live as long as the
# model; this process holds one model.
_LAYER_MODULES: dict[tuple[str, int], object] = {}
# Active group definitions populated as layers are constructed.
_GROUPS: dict[str, "Group"] = {}
_LOCK = threading.Lock()


@dataclass(frozen=True)
class Group:
    """One layout group: a model's MoE layers over `shards` ranks.

    `slots` is the physical slots per rank: the `active` ones the rank lists
    fill, plus `staging` spare slots a dynamic re-layout stages the next
    generation's experts into (unreferenced by the table, never routed to,
    skipped by the kernel's visit list). `replica_cap` fixes the slot table's
    width so a re-layout can give an expert more copies without changing a
    shape; None takes the most copies any expert of the layer has.
    """

    experts: int
    shards: int
    slots: int
    layers: dict[int, tuple[tuple[int, ...], ...]]
    staging: int = 0
    replica_cap: int | None = None

    @property
    def active(self) -> int:
        """Slots per rank the layout fills."""
        return self.slots - self.staging

    @property
    def physical(self) -> int:
        """Physical slots over the mesh, the kernel's `e_total`."""
        return self.shards * self.slots

    def ranks(self, layer: int) -> tuple[tuple[int, ...], ...] | None:
        """Per rank, the global experts it holds in slot order; None if unlisted."""
        return self.layers.get(layer)

    def slot_table(self, layer: int) -> list[list[int]]:
        """Row e: the slots r * slots + j holding expert e, padded with -1 to
        the most replicas any expert of the layer has."""
        held_at: list[list[int]] = [[] for _ in range(self.experts)]
        for r, held in enumerate(self.layers[layer]):
            for j, e in enumerate(held):
                held_at[e].append(r * self.slots + j)
        width = max(len(h) for h in held_at)
        if self.replica_cap is not None:
            width = max(width, self.replica_cap)
        return [h + [-1] * (width - len(h)) for h in held_at]

    def expert_map(self, layer: int, rank: int) -> list[int]:
        """Global expert -> this rank's local slot, or -1."""
        table = [-1] * self.experts
        for j, e in enumerate(self.layers[layer][rank]):
            table[e] = j
        return table


@dataclass(frozen=True)
class Placement:
    target: Group
    draft: Group | None = None

    def group(self, name: str) -> Group | None:
        return self.target if name == TARGET else self.draft

    @property
    def experts(self) -> int:
        return self.target.experts

    @property
    def shards(self) -> int:
        return self.target.shards

    def slots(self, layer: int) -> int:
        return self.target.slots

    def ranks(self, layer: int):
        return self.target.ranks(layer)

    def slot_table(self, layer: int):
        return self.target.slot_table(layer)


def default_ranks(
    experts: int, shards: int, extra_slots: int = DEFAULT_EXTRA_SLOTS
) -> tuple[tuple[int, ...], ...]:
    """Contiguous block per rank plus the first `extra_slots` expert(s) of the
    following rank(s), wrapping modulo `experts`."""
    if shards <= 0 or experts <= 0 or experts % shards != 0:
        raise ValueError(
            f"expert placement: {experts} experts do not divide over {shards} ranks"
        )
    per_shard = experts // shards
    if extra_slots < 0 or per_shard + extra_slots > experts:
        raise ValueError(
            f"expert placement: invalid extra_slots={extra_slots} for "
            f"{experts} experts over {shards} ranks"
        )
    width = per_shard + extra_slots
    return tuple(
        tuple((r * per_shard + j) % experts for j in range(width))
        for r in range(shards)
    )


def _parse_group(table: dict, source: str, what: str, shards: int | None) -> Group:
    experts = int(table["experts"])
    if shards is None:
        shards = int(table["shards"])
    if experts % shards:
        raise ValueError(
            f"expert layout {source}: {what} {experts} experts do not divide "
            f"over {shards} ranks"
        )
    layers = {}
    slots = int(table["slots"]) if "slots" in table else None
    staging = int(table.get("staging", 0))
    replica_cap = table.get("replica_cap")
    replica_cap = None if replica_cap is None else int(replica_cap)
    if staging < 0 or (replica_cap is not None and replica_cap < 1):
        raise ValueError(
            f"expert layout {source}: {what} staging must be >= 0 and "
            f"replica_cap >= 1; got {staging} and {replica_cap}"
        )
    most_copies = 1
    for key, ranks in table["layers"].items():
        width = len(ranks[0]) if ranks else 0
        if slots is None:
            slots = width
        if (
            len(ranks) != shards
            or width != slots
            or slots < experts // shards
            or any(len(held) != slots for held in ranks)
        ):
            widths = sorted({len(held) for held in ranks})
            raise ValueError(
                f"expert layout {source}: {what} layer {key} must list {shards} "
                f"ranks of {slots if slots is not None else 'one'} slots each, "
                f"at least {experts // shards}; found {len(ranks)} ranks of "
                f"widths {widths}"
            )
        ranks = tuple(tuple(int(e) for e in held) for held in ranks)
        if any(len(set(held)) != len(held) for held in ranks):
            raise ValueError(
                f"expert layout {source}: {what} layer {key} holds an expert "
                "twice on one rank"
            )
        if sorted({e for held in ranks for e in held}) != list(range(experts)):
            raise ValueError(
                f"expert layout {source}: {what} layer {key} must hold every "
                f"one of the {experts} experts"
            )
        copies = [0] * experts
        for held in ranks:
            for e in held:
                copies[e] += 1
        most_copies = max(most_copies, max(copies))
        layers[int(key)] = ranks
    if slots is None:
        slots = experts // shards
    if replica_cap is not None and most_copies > replica_cap:
        raise ValueError(
            f"expert layout {source}: {what} gives an expert {most_copies} "
            f"copies, above replica_cap {replica_cap}"
        )
    return Group(
        experts=experts,
        shards=shards,
        slots=slots + staging,
        layers=layers,
        staging=staging,
        replica_cap=replica_cap,
    )


def parse(table: dict, source: str = "<dict>") -> Placement:
    """Validate an in-memory layout dictionary: every expert held, no rank
    holds one twice, equal slots on every rank of a layer."""
    target = _parse_group(table, source, TARGET, None)
    draft = None
    if table.get("draft") is not None:
        draft = _parse_group(table["draft"], source, DRAFT, target.shards)
    return Placement(target=target, draft=draft)


def current(vllm_config=None) -> Placement | None:
    """The process's active dynamic placement, or None when disabled."""
    del vllm_config
    with _LOCK:
        target = _GROUPS.get(TARGET)
        if target is None:
            return None
        return Placement(target=target, draft=_GROUPS.get(DRAFT))


def layer_index(name: str) -> int | None:
    """The decoder layer a module prefix or checkpoint tensor name belongs to."""
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _runner_group() -> tuple[str, int]:
    """Which group the model being built belongs to, and the module index
    its first MoE layer carries."""
    try:
        from vllm.config import get_current_vllm_config

        model_config = get_current_vllm_config().model_config
        runner_type = model_config.runner_type
    except Exception:  # noqa: BLE001 - no config: the target
        return TARGET, 0
    if runner_type != "draft":
        return TARGET, 0
    base = int(getattr(model_config.hf_config, "num_hidden_layers", 0) or 0)
    return DRAFT, base


def has_applied_layers() -> bool:
    return bool(_APPLIED_LAYERS)


def fetch_filter_ids(name: str, ep_rank: int) -> frozenset[int] | None:
    """The experts the fetch-time filter keeps for this tensor's layer.

    None defers to the default contiguous filter: either there is no
    layout for the layer, or no MoE method applied one to it.
    """
    drafter = _DRAFTER_TENSOR_RE.match(name)
    if drafter is not None:
        group_name, layer = DRAFT, int(drafter.group(1))
    else:
        group_name, layer = TARGET, layer_index(name)
    if layer is None or (group_name, layer) not in _APPLIED_LAYERS:
        return None
    placement = current()
    group = placement.group(group_name) if placement is not None else None
    if group is None:
        return None
    ranks = group.ranks(layer)
    if ranks is None:
        return None
    return frozenset(ranks[ep_rank])


def apply_to_layer(layer) -> bool:
    """Initialize a RoutedExperts layer with the default Dynamic EPLB layout
    (contiguous block + next rank's first expert + staging slots).

    Must run before the weights load (from the MoE method's create_weights).
    Registers the slot table the fused EP op takes. Returns whether the layer
    is placed.
    """
    import torch

    if not envs.TPU_MOE_EPLB_DYNAMIC:
        return False
    group_name, _base = _runner_group()
    if group_name != TARGET:
        return False
    layer_idx = layer_index(layer.layer_name)
    if layer_idx is None:
        return False
    manager = layer.expert_map_manager
    experts = int(manager.global_num_experts)
    shards = int(manager.ep_size)
    if shards <= 1 or experts <= 0 or experts % shards != 0:
        return False
    if manager.num_fused_shared_experts or manager.expert_mask is not None:
        return False

    ranks = default_ranks(experts, shards, extra_slots=DEFAULT_EXTRA_SLOTS)
    active_slots = len(ranks[0])
    total_slots = active_slots + DEFAULT_STAGING
    with _LOCK:
        existing = _GROUPS.get(group_name)
        layers_dict = dict(existing.layers) if existing is not None else {}
        if existing is not None and (
            existing.experts != experts or existing.shards != shards
        ):
            raise ValueError(
                f"expert placement ({group_name}) is for {existing.shards} "
                f"ranks x {existing.experts} experts; {layer.layer_name} has "
                f"{shards} x {experts}"
            )
        layers_dict[layer_idx] = ranks
        group = Group(
            experts=experts,
            shards=shards,
            slots=total_slots,
            layers=layers_dict,
            staging=DEFAULT_STAGING,
            replica_cap=shards,
        )
        _GROUPS[group_name] = group

    rank = manager.ep_rank
    expert_map = torch.tensor(group.expert_map(layer_idx, rank), dtype=torch.int32)
    device = manager.expert_map.device
    manager._expert_map = expert_map.to(device)
    manager._local_num_experts = group.slots
    layer.update_expert_map_info()
    layer.register_buffer(
        EXPERT_SLOT_ATTR,
        torch.tensor(group.slot_table(layer_idx), dtype=torch.int32, device=device),
        persistent=False,
    )
    setattr(layer, _GROUP_ATTR, group_name)
    setattr(layer, LAYOUT_KEY_ATTR, (group_name, layer_idx))
    with _LOCK:
        _APPLIED_LAYERS.add((group_name, layer_idx))
        _LAYER_MODULES[(group_name, layer_idx)] = layer
    logger.info_once(
        "Dynamic EPLB initial layout applied (%s group): rank %d of %d, "
        "%d slots per rank (%d active + %d staging); first placed layer holds %s",
        group_name,
        rank,
        group.shards,
        group.slots,
        group.active,
        group.staging,
        ranks[rank][:6],
    )
    return True


def expert_slot_table(layer):
    """The [experts, replicas] i32 slot table of a placed layer, or None."""
    return getattr(layer, EXPERT_SLOT_ATTR, None)


def applied_layers() -> dict[tuple[str, int], object]:
    """(group, layer index) -> placed layer module, for every applied layer."""
    with _LOCK:
        return dict(_LAYER_MODULES)


def layout_key(layer) -> tuple[str, int] | None:
    """The (group, layer index) a placed layer follows, or None."""
    return getattr(layer, LAYOUT_KEY_ATTR, None)


def install_table(layer, table) -> None:
    """Write new slot-table VALUES into a placed layer's table buffer.

    The buffer's shape (experts x replica cap) is a compiled-program input and
    does not change; the values do, between two steps, once every rank has the
    experts the new table names in the slots it names (expert_relayout.py).
    """
    import torch

    buf = expert_slot_table(layer)
    if buf is None:
        raise RuntimeError(f"{getattr(layer, 'layer_name', layer)} has no slot table")
    new = torch.as_tensor(table, dtype=torch.int32)
    if tuple(new.shape) != tuple(buf.shape):
        raise ValueError(
            f"slot table {tuple(new.shape)} does not match the layer's "
            f"{tuple(buf.shape)}; the replica cap is fixed at load"
        )
    buf.copy_(new.to(buf.device))


def physical_experts(layer) -> int | None:
    """The placed layer's physical slot count over the mesh, or None."""
    table = expert_slot_table(layer)
    if table is None:
        return None
    placement = current()
    if placement is None:
        return None
    group = placement.group(getattr(layer, _GROUP_ATTR, TARGET))
    return group.physical if group is not None else None


def check_layer_served(layer, fused_armed: bool) -> None:
    """A placed layer's weights sit in slot order only the fused op undoes."""
    if expert_slot_table(layer) is not None and not fused_armed:
        raise RuntimeError(
            f"expert layout covers {layer.layer_name}, but the fused EP "
            "MoE kernel is not armed for it; the grouped-matmul path reads "
            "contiguous expert blocks and would run the wrong experts"
        )
