"""
apply_parallelism — replaces nn.Linear with parallel layers based on a spec.

Usage:
    spec = ParallelSpec({
        "blocks.*.self_attn.qkv_proj":      ColumnParallelSpec(sequence_parallel=True),
        "blocks.*.self_attn.output_proj":   RowParallelSpec(),
        "blocks.*.cross_attn.q_proj":       ColumnParallelSpec(sequence_parallel=True),
        "blocks.*.cross_attn.k_proj":       ColumnParallelSpec(sequence_parallel=False),
        "blocks.*.cross_attn.v_proj":       ColumnParallelSpec(sequence_parallel=False),
        "blocks.*.cross_attn.output_proj":  RowParallelSpec(),
        "blocks.*.mlp.gate_up_proj":        ColumnParallelSpec(sequence_parallel=True),
        "blocks.*.mlp.down_proj":           RowParallelSpec(),
    })
    model = apply_parallelism(model, spec, config, groups)

Pattern syntax: use * to match any single path component (e.g. list indices).
"blocks.*.self_attn.qkv_proj" matches blocks.0.self_attn.qkv_proj, blocks.1..., etc.

After apply_parallelism:
  - Matched nn.Linear modules are replaced with ColumnParallelLinear or RowParallelLinear
  - Weights are sharded in-place (each rank holds its slice only)
  - model._wdp_groups and model._wdp_config are set for use in forward()
  - Call sync_replicated_grads(model, groups.tp) after every backward()

sync_replicated_grads:
  All-reduces gradients of non-sharded parameters (LayerNorm, AdaLN, embedders, etc.)
  across TP ranks to keep replicated weights identical. Must be called after backward.
"""

from dataclasses import dataclass
from typing import Optional
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist

from .config import ParallelConfig
from .initialize import ProcessGroups
from .layers import ColumnParallelLinear, RowParallelLinear


# ---------------------------------------------------------------------------
# Spec types
# ---------------------------------------------------------------------------

@dataclass
class ColumnParallelSpec:
    """Replace nn.Linear with ColumnParallelLinear.

    sequence_parallel: if True, all-gather input along seq_dim before matmul.
                       Set False for cross-attn K/V projections where the context
                       tensor is already replicated across ranks.
    seq_dim:           which tensor dimension holds the sequence tokens.
                       0 (default) → Megatron-style (S, B, D)
                       1           → batch-first    (B, S, D)  ← Anima, HuggingFace
    """
    sequence_parallel:  bool = True
    seq_dim:            int  = 0
    packed_parts:       Optional[int] = None
    allow_padding:      bool = False
    padding_multiple:   int  = 1


@dataclass
class PackedColumnParallelSpec(ColumnParallelSpec):
    """Column-parallel spec for packed projections such as QKV or SwiGLU.

    packed_parts is the number of equal logical chunks in the output feature
    dimension. Each part is sharded independently so a later tensor.chunk()
    still sees local slices of every logical part.
    """
    packed_parts: int = 2


@dataclass
class RowParallelSpec:
    """Replace nn.Linear with RowParallelLinear.

    sequence_parallel: if True, reduce-scatter output along seq_dim after matmul.
                       Set False if the output should stay fully replicated.
    seq_dim:           which tensor dimension holds the sequence tokens.
                       0 (default) → Megatron-style (S, B, D)
                       1           → batch-first    (B, S, D)  ← Anima, HuggingFace
    """
    sequence_parallel: bool = True
    seq_dim:           int  = 0
    allow_padding:     bool = False
    padding_multiple:  int  = 1


class ParallelSpec:
    """
    Maps module path patterns to parallel layer specs.

    Pattern rules:
      - Dot-separated module path, e.g. "blocks.0.self_attn.qkv_proj"
      - Use * to match any single component: "blocks.*.self_attn.qkv_proj"
      - No regex — * is the only wildcard, matches exactly one component
    """

    def __init__(self, entries: dict):
        self.entries = entries   # {pattern_str: ColumnParallelSpec | RowParallelSpec}

    def match(self, path: str):
        """Return the spec for path, or None if unmatched."""
        for pattern, spec in self.entries.items():
            if _match_path(pattern, path):
                return spec
        return None


def _match_path(pattern: str, path: str) -> bool:
    pp = pattern.split(".")
    pq = path.split(".")
    if len(pp) != len(pq):
        return False
    return all(a == b or a == "*" for a, b in zip(pp, pq))


def _resolve_parent_attr(model: nn.Module, name: str):
    if "." not in name:
        return model, name
    parent_path, attr = name.rsplit(".", 1)
    return model.get_submodule(parent_path), attr


def _effective_sequence_parallel(layer_spec, config_sp: bool, warn_state: list[bool]) -> bool:
    requested = getattr(layer_spec, "sequence_parallel", False)
    if requested and not config_sp and not warn_state[0]:
        warnings.warn(
            "sequence_parallel=True in spec while config.sp=False; "
            "falling back to TP-only collectives for matched layers."
        )
        warn_state[0] = True
    return requested and config_sp


def _build_parallel_linear(
    module: nn.Linear,
    layer_spec,
    rank: int,
    world_size: int,
    group: dist.ProcessGroup,
    sequence_parallel: bool,
):
    if isinstance(layer_spec, ColumnParallelSpec):
        packed_parts = layer_spec.packed_parts
        new_layer = ColumnParallelLinear.from_linear(
            module,
            rank,
            world_size,
            group,
            sequence_parallel=sequence_parallel,
            seq_dim=layer_spec.seq_dim,
            packed_parts=packed_parts,
            allow_padding=layer_spec.allow_padding,
            padding_multiple=layer_spec.padding_multiple,
        )
        shard_kind = "column" if packed_parts is None else f"packed-column({packed_parts})"
        if layer_spec.allow_padding:
            shard_kind = f"padded-{shard_kind}"
        return new_layer, shard_kind

    if isinstance(layer_spec, RowParallelSpec):
        new_layer = RowParallelLinear.from_linear(
            module,
            rank,
            world_size,
            group,
            sequence_parallel=sequence_parallel,
            seq_dim=layer_spec.seq_dim,
            allow_padding=layer_spec.allow_padding,
            padding_multiple=layer_spec.padding_multiple,
        )
        shard_kind = "padded-row" if layer_spec.allow_padding else "row"
        return new_layer, shard_kind

    return None, None


def _mark_tp_sharding_metadata(model: nn.Module, rank: int, world_size: int) -> None:
    """Tag every TP-sharded parameter with reconstruction metadata.

    ``_tp_sharded`` gates sync_replicated_grads (skip these params) and the
    LoRA/FFT TP grad-norm clipping (all-reduce-sum these params' contribution
    instead of counting it once). The rest of the ``_tp_*`` fields let a
    TP-aware optimizer (see muon_optimizer.TPMuonWithAuxAdam) reconstruct the
    full unsharded weight for an operation that must see the whole matrix
    (Newton-Schulz orthogonalization), then re-shard the result — using the
    same shard/pad/merge helpers apply_parallelism used to build the shard in
    the first place (layers._shard_colwise / _shard_packed_colwise /
    _shard_rowwise for re-sharding; layers.merge_column_shards /
    merge_row_shards for reconstruction).

    ``rank``/``world_size`` come from the same ``ProcessGroups`` the caller
    already resolved for sharding (not re-derived via dist.get_rank(group=...)
    here) — every ColumnParallelLinear/RowParallelLinear in a given
    apply_parallelism() call shares one TP group, and re-deriving per module
    would require a live default process group even in contexts (e.g. tests
    with a fake group double) that only need the metadata, not a collective.

    RowParallel bias is replicated across TP ranks and must be synced.
    """
    for module in model.modules():
        if isinstance(module, ColumnParallelLinear):
            shard_kind = "col" if module.packed_parts is None else "packed_col"
            for p in module.parameters():
                p._tp_sharded              = True
                p._tp_shard_kind           = shard_kind
                p._tp_rank                 = rank
                p._tp_world_size           = world_size
                p._tp_packed_parts         = module.packed_parts
                p._tp_original_part_size   = module.original_part_size
                p._tp_padded_part_size     = module.padded_part_size
                p._tp_original_out_features = module.original_out_features
                p._tp_original_in_features  = module.original_in_features
                p._tp_allow_padding        = module.allow_padding
                p._tp_padding_multiple     = module.padding_multiple
            continue
        if isinstance(module, RowParallelLinear):
            module.weight._tp_sharded              = True
            module.weight._tp_shard_kind            = "row"
            module.weight._tp_rank                  = rank
            module.weight._tp_world_size            = world_size
            module.weight._tp_packed_parts          = None
            module.weight._tp_original_part_size    = None
            module.weight._tp_padded_part_size      = None
            module.weight._tp_original_out_features = module.original_out_features
            module.weight._tp_original_in_features  = module.original_in_features
            module.weight._tp_allow_padding         = module.allow_padding
            module.weight._tp_padding_multiple      = module.padding_multiple
            if module.bias is not None:
                module.bias._tp_sharded = False
                if module.sequence_parallel:
                    module.bias._tp_partial_grad = True


# ---------------------------------------------------------------------------
# apply_parallelism
# ---------------------------------------------------------------------------

def apply_parallelism(
    model:   nn.Module,
    spec:    ParallelSpec,
    config:  ParallelConfig,
    groups:  ProcessGroups,
    debug:   bool = False,
) -> nn.Module:
    """
    Replace matched nn.Linear modules with parallel equivalents and shard weights.

    Steps:
      1. Walk model.named_modules()
      2. For each module matching a spec entry and isinstance nn.Linear:
         - Create ColumnParallelLinear or RowParallelLinear from the existing weight
         - Replace via setattr on parent module
      3. Tag sharded parameters with ._tp_sharded = True
      4. Set model._wdp_groups and model._wdp_config

    Returns the modified model (also modifies in-place).
    """
    if not config.tp:
        return model

    rank       = groups.tp_rank
    world_size = groups.tp_size
    group      = groups.tp

    # Collect (parent, attr_name, replacement) — don't mutate during iteration
    replacements = []
    replacement_debug = []
    warned_sp_disabled = [False]
    for name, module in model.named_modules():
        layer_spec = spec.match(name)
        if layer_spec is None:
            continue
        if not isinstance(module, nn.Linear):
            continue

        # Split "a.b.c" → parent path "a.b", attr "c"
        parent, attr = _resolve_parent_attr(model, name)

        sequence_parallel = _effective_sequence_parallel(
            layer_spec, config.sp, warned_sp_disabled
        )

        try:
            new_layer, shard_kind = _build_parallel_linear(
                module, layer_spec, rank, world_size, group, sequence_parallel
            )
            if new_layer is None:
                continue
        except ValueError as e:
            raise ValueError(f"{name}: {e}") from e

        replacements.append((parent, attr, new_layer))
        replacement_debug.append((name, shard_kind, sequence_parallel, layer_spec.seq_dim))

    for parent, attr, new_layer in replacements:
        setattr(parent, attr, new_layer)

    # Tag sharded parameters so sync_replicated_grads can skip them.
    _mark_tp_sharding_metadata(model, rank, world_size)

    # Store config/groups on the model so forward() can access SP boundaries
    model._wdp_config = config
    model._wdp_groups = groups

    if debug and rank == 0:
        print("  [wd_parallel] sharding map:")
        for name, shard_kind, seqp, seq_dim in replacement_debug:
            print(
                f"    - {name}: {shard_kind}  "
                f"sequence_parallel={seqp}  seq_dim={seq_dim}"
            )

    return model


# ---------------------------------------------------------------------------
# set_wgrad_gather_reuse
# ---------------------------------------------------------------------------

def set_wgrad_gather_reuse(model: nn.Module, enabled: bool) -> int:
    """Toggle forward-gather reuse for the wgrad computation on every
    ColumnParallelLinear in ``model`` (see ColumnParallelLinear.forward and
    collectives._ColumnLinearFwdBwd).

    When enabled, the SP fused backward skips re-all-gathering the input
    shard for the weight gradient and reuses the tensor the forward pass
    already gathered — one fewer collective per trainable column-parallel
    layer per backward. Only takes effect for layers whose weight requires
    grad and whose TP group size > 1 (frozen/LoRA-base and single-GPU layers
    are unaffected either way).

    Costs extra memory to keep the full-sequence activation alive across the
    backward pass; safe to enable unconditionally under gradient
    checkpointing (that activation is recomputed and freed immediately
    after use), but should be left off otherwise unless VRAM allows it.

    Returns the number of ColumnParallelLinear modules touched.
    """
    n = 0
    for module in model.modules():
        if isinstance(module, ColumnParallelLinear):
            module.reuse_fwd_gather_for_wgrad = bool(enabled)
            n += 1
    return n


# ---------------------------------------------------------------------------
# sync_replicated_grads
# ---------------------------------------------------------------------------

def _coalesced_all_reduce_by_dtype(
    entries: list, group: dist.ProcessGroup, world_size: int, *, divide: bool,
) -> None:
    """All-reduce (param, grad) pairs, one flat collective per distinct grad dtype.

    torch.cat requires matching dtypes, so mixed-precision grads (e.g. under
    --muon_fp32_sensitive, which upcasts adaln_modulation/q_norm/k_norm to
    fp32 while the rest of the model trains in bf16) are bucketed separately
    instead of crashing. Same-dtype callers (the common case) still get
    exactly one collective, unchanged from before.
    """
    buckets: dict = {}
    for p, g in entries:
        buckets.setdefault(g.dtype, []).append((p, g))
    for bucket in buckets.values():
        flat = torch.cat([g.contiguous().view(-1) for _, g in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
        if divide:
            flat.div_(world_size)
        offset = 0
        for p, g in bucket:
            n = g.numel()
            g.copy_(flat[offset: offset + n].view_as(g))
            offset += n


def sync_replicated_grads(model: nn.Module, group: dist.ProcessGroup) -> None:
    """
    All-reduce gradients of replicated (non-TP-sharded) parameters.

    Must be called after every loss.backward(). Without this, non-TP params
    (LayerNorm, AdaLN projections, embedders, final proj) diverge between
    ranks after the first optimizer step.

    Dense params are coalesced into one flat all-reduce per grad dtype
    (instead of one collective per parameter). This is critical for backends
    with high per-collective overhead (e.g. cuda_direct on Windows) where N
    individual all-reduces x barrier_cost dominates comm time.

    Sparse and partial-grad params are handled separately since they need
    different reduce semantics or can't be safely flattened.
    """
    if group.size() == 1:
        return

    world_size = group.size()

    dense_mean: list[tuple] = []   # (param, grad) — all-reduce then div
    dense_sum:  list[tuple] = []   # (param, grad) — all-reduce, no div (_tp_partial_grad)
    sparse_infos: list[tuple] = [] # (param, dense_grad, partial)

    for p in model.parameters():
        grad = p.grad
        if grad is None or getattr(p, "_tp_sharded", False):
            continue
        partial = getattr(p, "_tp_partial_grad", False)
        if grad.is_sparse:
            sparse_infos.append((p, grad.coalesce().to_dense(), partial))
        elif partial:
            dense_sum.append((p, grad))
        else:
            dense_mean.append((p, grad))

    # Coalesced all-reduce for mean params (the common case).
    if dense_mean:
        _coalesced_all_reduce_by_dtype(dense_mean, group, world_size, divide=True)

    # Coalesced all-reduce for partial-grad params (sum semantics, no div).
    if dense_sum:
        _coalesced_all_reduce_by_dtype(dense_sum, group, world_size, divide=False)

    # Individual all-reduce for sparse grads (rare; can't be safely flattened).
    for p, dense_grad, partial in sparse_infos:
        dist.all_reduce(dense_grad, op=dist.ReduceOp.SUM, group=group)
        if not partial:
            dense_grad.div_(world_size)
        p.grad = dense_grad.to_sparse_coo().coalesce()
