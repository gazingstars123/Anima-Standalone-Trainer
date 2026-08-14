"""
Parallel linear layers for TP+SP.

ColumnParallelLinear — shards output features (columns) across TP ranks.
  Weight: [in, out] → [in, out/tp] per rank
  Input:  replicated or SP-sharded along seq_dim
  Output: full sequence along seq_dim, feature-sharded (D_out/tp)
  SP forward:  all-gather along seq_dim before matmul (sequence_parallel=True)
  SP backward: reduce-scatter input grad (symmetric with gather)

RowParallelLinear — shards input features (rows) across TP ranks.
  Weight: [in, out] → [in/tp, out] per rank
  Input:  feature-sharded (D_in/tp), full sequence along seq_dim
  Output: SP-sharded along seq_dim (D_out) in SP mode, replicated in TP-only mode
  SP forward:  reduce-scatter along seq_dim after matmul (fuses TP all-reduce + SP scatter)
  SP backward: all-gather grad (symmetric with reduce-scatter)

seq_dim controls which tensor dimension holds the sequence tokens:
  seq_dim=0 (default)  Megatron-style (S, B, D) — fast pre-buffered path
  seq_dim=1            batch-first    (B, S, D) — Anima, HuggingFace models
  seq_dim=N            arbitrary      any model where sequence is at dim N

Both layers expose a from_linear() class method for zero-copy construction
from an existing nn.Linear weight (slices the weight in-place without copying
the full tensor across ranks).
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional

from .collectives import (
    gather_from_sp_region,
    reduce_scatter_to_sp_region,
    copy_to_tp_region,
    copy_to_tp_region_no_input_grad,
    _split_along_first_dim,
    _split_along_dim,
    _ColumnLinearFwdBwd,
    _RowLinearFwdBwd,
)


def _ceil_to_multiple(size: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be > 0")
    return ((size + multiple - 1) // multiple) * multiple


def _padded_size_for_tp(size: int, world_size: int, padding_multiple: int = 1) -> int:
    return _ceil_to_multiple(size, world_size * padding_multiple)


def _pad_along_dim(tensor: torch.Tensor, dim: int, padded_size: int) -> torch.Tensor:
    if tensor.size(dim) == padded_size:
        return tensor
    if tensor.size(dim) > padded_size:
        raise ValueError(
            f"cannot pad dim {dim} from {tensor.size(dim)} down to {padded_size}"
        )
    out_shape = list(tensor.shape)
    out_shape[dim] = padded_size
    padded = tensor.new_zeros(out_shape)
    index = [slice(None)] * tensor.ndim
    index[dim] = slice(0, tensor.size(dim))
    padded[tuple(index)] = tensor
    return padded


def _shard_colwise(
    weight: torch.Tensor,
    rank: int,
    world_size: int,
    *,
    allow_padding: bool = False,
    padding_multiple: int = 1,
) -> torch.Tensor:
    """Slice output-feature dim: weight[out_start:out_end, :]."""
    out = weight.size(0)
    if allow_padding:
        padded_out = _padded_size_for_tp(out, world_size, padding_multiple)
        weight = _pad_along_dim(weight, 0, padded_out)
    elif out % world_size != 0:
        raise ValueError(
            f"output dim {out} is not divisible by tp_size={world_size}"
        )
    chunk = weight.size(0) // world_size
    return weight[rank * chunk : (rank + 1) * chunk].contiguous()


def _shard_bias_colwise(
    bias: torch.Tensor,
    rank: int,
    world_size: int,
    *,
    allow_padding: bool = False,
    padding_multiple: int = 1,
) -> torch.Tensor:
    out = bias.size(0)
    if allow_padding:
        padded_out = _padded_size_for_tp(out, world_size, padding_multiple)
        bias = _pad_along_dim(bias, 0, padded_out)
    elif out % world_size != 0:
        raise ValueError(
            f"bias dim {out} is not divisible by tp_size={world_size}"
        )
    chunk = bias.size(0) // world_size
    return bias[rank * chunk : (rank + 1) * chunk].contiguous()


def _shard_packed_colwise(
    weight: torch.Tensor,
    rank: int,
    world_size: int,
    packed_parts: int,
    *,
    allow_padding: bool = False,
    padding_multiple: int = 1,
) -> torch.Tensor:
    """Slice each packed output part independently, then concatenate local shards."""
    out = weight.size(0)
    if out % packed_parts != 0:
        raise ValueError(
            f"output dim {out} is not divisible by packed_parts={packed_parts}"
        )
    part_size = out // packed_parts
    if part_size % world_size != 0 and not allow_padding:
        raise ValueError(
            f"packed part size {part_size} is not divisible by tp_size={world_size}"
        )
    padded_part_size = (
        _padded_size_for_tp(part_size, world_size, padding_multiple)
        if allow_padding else part_size
    )
    chunk = padded_part_size // world_size
    shards = []
    for part in range(packed_parts):
        if allow_padding and padded_part_size != part_size:
            part_weight = weight[part * part_size : (part + 1) * part_size]
            part_weight = _pad_along_dim(part_weight, 0, padded_part_size)
            shards.append(part_weight[rank * chunk : (rank + 1) * chunk])
        else:
            start = part * part_size + rank * chunk
            end = start + chunk
            shards.append(weight[start:end])
    return torch.cat(shards, dim=0).contiguous()


def _shard_packed_bias(
    bias: torch.Tensor,
    rank: int,
    world_size: int,
    packed_parts: int,
    *,
    allow_padding: bool = False,
    padding_multiple: int = 1,
) -> torch.Tensor:
    """Slice a packed bias with the same layout as _shard_packed_colwise."""
    out = bias.size(0)
    if out % packed_parts != 0:
        raise ValueError(
            f"bias dim {out} is not divisible by packed_parts={packed_parts}"
        )
    part_size = out // packed_parts
    if part_size % world_size != 0 and not allow_padding:
        raise ValueError(
            f"packed bias part size {part_size} is not divisible by tp_size={world_size}"
        )
    padded_part_size = (
        _padded_size_for_tp(part_size, world_size, padding_multiple)
        if allow_padding else part_size
    )
    chunk = padded_part_size // world_size
    shards = []
    for part in range(packed_parts):
        if allow_padding and padded_part_size != part_size:
            part_bias = bias[part * part_size : (part + 1) * part_size]
            part_bias = _pad_along_dim(part_bias, 0, padded_part_size)
            shards.append(part_bias[rank * chunk : (rank + 1) * chunk])
        else:
            start = part * part_size + rank * chunk
            end = start + chunk
            shards.append(bias[start:end])
    return torch.cat(shards, dim=0).contiguous()


def _shard_rowwise(
    weight: torch.Tensor,
    rank: int,
    world_size: int,
    *,
    allow_padding: bool = False,
    padding_multiple: int = 1,
) -> torch.Tensor:
    """Slice input-feature dim: weight[:, in_start:in_end]."""
    in_ = weight.size(1)
    if allow_padding:
        padded_in = _padded_size_for_tp(in_, world_size, padding_multiple)
        weight = _pad_along_dim(weight, 1, padded_in)
    elif in_ % world_size != 0:
        raise ValueError(
            f"input dim {in_} is not divisible by tp_size={world_size}"
        )
    chunk = weight.size(1) // world_size
    return weight[:, rank * chunk : (rank + 1) * chunk].contiguous()


def trim_padded_features(
    tensor: torch.Tensor,
    original_size: int,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Trim a padded feature dimension back to its original size."""
    dim = dim % tensor.ndim
    if tensor.size(dim) < original_size:
        raise ValueError(
            f"cannot trim dim {dim} of size {tensor.size(dim)} to larger original_size={original_size}"
        )
    index = [slice(None)] * tensor.ndim
    index[dim] = slice(0, original_size)
    return tensor[tuple(index)].contiguous()


def trim_packed_padded_features(
    tensor: torch.Tensor,
    *,
    packed_parts: int,
    original_part_size: int,
    padded_part_size: int,
    dim: int = -1,
) -> torch.Tensor:
    """Trim each packed padded part, preserving Q/K/V or gate/up boundaries."""
    dim = dim % tensor.ndim
    expected = packed_parts * padded_part_size
    if tensor.size(dim) != expected:
        raise ValueError(
            f"packed dim size {tensor.size(dim)} does not match "
            f"packed_parts*padded_part_size={expected}"
        )
    parts = tensor.split(padded_part_size, dim=dim)
    trimmed = [
        trim_padded_features(part, original_part_size, dim=dim)
        for part in parts
    ]
    return torch.cat(trimmed, dim=dim).contiguous()


def merge_column_shards(
    shards: list[torch.Tensor],
    *,
    original_out_features: int,
    dim: int = 0,
    packed_parts: Optional[int] = None,
    original_part_size: Optional[int] = None,
    padded_part_size: Optional[int] = None,
) -> torch.Tensor:
    """Concatenate column shards and remove padded output rows/features."""
    if packed_parts is None:
        merged = torch.cat(shards, dim=dim)
        return trim_padded_features(merged, original_out_features, dim=dim)
    if original_part_size is None or padded_part_size is None:
        raise ValueError("packed reconstruction requires original_part_size and padded_part_size")
    dim = dim % shards[0].ndim
    local_part_size = padded_part_size // len(shards)
    reconstructed_parts = []
    for part_idx in range(packed_parts):
        local_parts = []
        for shard in shards:
            if shard.size(dim) != packed_parts * local_part_size:
                raise ValueError(
                    f"packed shard dim size {shard.size(dim)} does not match "
                    f"packed_parts*local_part_size={packed_parts * local_part_size}"
                )
            start = part_idx * local_part_size
            index = [slice(None)] * shard.ndim
            index[dim] = slice(start, start + local_part_size)
            local_parts.append(shard[tuple(index)])
        full_part = torch.cat(local_parts, dim=dim)
        reconstructed_parts.append(
            trim_padded_features(full_part, original_part_size, dim=dim)
        )
    return torch.cat(reconstructed_parts, dim=dim).contiguous()


def merge_row_shards(
    shards: list[torch.Tensor],
    *,
    original_in_features: int,
    dim: int = 1,
) -> torch.Tensor:
    """Concatenate row shards and remove padded input columns/features."""
    return trim_padded_features(
        torch.cat(shards, dim=dim),
        original_in_features,
        dim=dim,
    )


class ColumnParallelLinear(nn.Module):
    """
    Column-parallel linear layer.

    sequence_parallel=True  (default):
        Forward: all-gather sequence along seq_dim before matmul.
                 Input  (…, S/tp, …) → gathered (…, S, …) → output (…, S, …, D_out/tp)
        Backward: reduce-scatter gradient of input.

    sequence_parallel=False:
        Forward: plain matmul on whatever input is provided.
                 Use for cross-attn K/V projections where context is replicated.
        Backward: all-reduce input gradient (_CopyToTPRegion).

    seq_dim: which dimension holds the sequence tokens (default 0).
        seq_dim=0 → Megatron (S, B, D)
        seq_dim=1 → batch-first (B, S, D)  ← use this for Anima / HuggingFace models
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        bias:         bool = False,
        sequence_parallel: bool = True,
        seq_dim:      int  = 0,
    ):
        super().__init__()
        self.in_features       = in_features
        self.out_features      = out_features
        self.sequence_parallel = sequence_parallel
        self.seq_dim           = seq_dim
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None
        self._group: Optional[dist.ProcessGroup] = None
        self.skip_input_grad = False
        # When True, the SP fused backward reuses the forward's all-gathered
        # activation for the weight gradient instead of re-gathering it (one
        # fewer collective per call). See apply.set_wgrad_gather_reuse().
        self.reuse_fwd_gather_for_wgrad = False

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    @classmethod
    def from_linear(
        cls,
        linear:       nn.Linear,
        rank:         int,
        world_size:   int,
        group:        dist.ProcessGroup,
        sequence_parallel: bool = True,
        seq_dim:      int  = 0,
        packed_parts: Optional[int] = None,
        allow_padding: bool = False,
        padding_multiple: int = 1,
    ) -> "ColumnParallelLinear":
        """Create from an existing nn.Linear, slicing its weight colwise."""
        if packed_parts is not None:
            if packed_parts < 2:
                raise ValueError("packed_parts must be >= 2")
            if linear.out_features % packed_parts != 0:
                raise ValueError(
                    f"out_features={linear.out_features} is not divisible by "
                    f"packed_parts={packed_parts}"
                )
            part_size = linear.out_features // packed_parts
            if part_size % world_size != 0:
                if not allow_padding:
                    raise ValueError(
                        f"packed part size {part_size} is not divisible by "
                        f"tp_size={world_size}"
                    )
            padded_part_size = (
                _padded_size_for_tp(part_size, world_size, padding_multiple)
                if allow_padding else part_size
            )
            padded_out_features = packed_parts * padded_part_size
            out_shard = packed_parts * (padded_part_size // world_size)
        else:
            if linear.out_features % world_size != 0:
                if not allow_padding:
                    raise ValueError(
                        f"out_features={linear.out_features} is not divisible by "
                        f"tp_size={world_size}"
                    )
            padded_out_features = (
                _padded_size_for_tp(linear.out_features, world_size, padding_multiple)
                if allow_padding else linear.out_features
            )
            out_shard = padded_out_features // world_size
        layer = cls(
            in_features       = linear.in_features,
            out_features      = out_shard,
            bias              = linear.bias is not None,
            sequence_parallel = sequence_parallel,
            seq_dim           = seq_dim,
        )
        if packed_parts is None:
            weight = _shard_colwise(
                linear.weight.data,
                rank,
                world_size,
                allow_padding=allow_padding,
                padding_multiple=padding_multiple,
            )
        else:
            weight = _shard_packed_colwise(
                linear.weight.data,
                rank,
                world_size,
                packed_parts,
                allow_padding=allow_padding,
                padding_multiple=padding_multiple,
            )
        layer.weight = nn.Parameter(weight)
        if linear.bias is not None:
            if packed_parts is None:
                bias = _shard_bias_colwise(
                    linear.bias.data,
                    rank,
                    world_size,
                    allow_padding=allow_padding,
                    padding_multiple=padding_multiple,
                )
            else:
                bias = _shard_packed_bias(
                    linear.bias.data,
                    rank,
                    world_size,
                    packed_parts,
                    allow_padding=allow_padding,
                    padding_multiple=padding_multiple,
                )
            layer.bias = nn.Parameter(bias)
        layer._group = group
        layer.original_in_features = linear.in_features
        layer.original_out_features = linear.out_features
        layer.padded_in_features = linear.in_features
        layer.padded_out_features = padded_out_features
        layer.allow_padding = allow_padding
        layer.padding_multiple = padding_multiple
        layer.packed_parts = packed_parts
        layer.skip_input_grad = False
        layer.original_part_size = (
            linear.out_features // packed_parts if packed_parts is not None else None
        )
        layer.padded_part_size = (
            padded_out_features // packed_parts if packed_parts is not None else None
        )
        layer.local_part_size = (
            layer.padded_part_size // world_size if packed_parts is not None else None
        )
        return layer

    def _prepare_tp_input(self, x: torch.Tensor) -> torch.Tensor:
        if self._group is not None and self._group.size() > 1:
            if self.sequence_parallel:
                return gather_from_sp_region(x, self._group, self.seq_dim)
            else:
                if self.skip_input_grad:
                    return copy_to_tp_region_no_input_grad(x, self._group)
                return copy_to_tp_region(x, self._group)
        return x

    def forward_from_prepared_input(self, x: torch.Tensor) -> torch.Tensor:
        adapter = getattr(self, "_tp_lora_adapter", None)
        if adapter is not None:
            return adapter.forward_from_prepared_input(x)
        return nn.functional.linear(x, self.weight, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapter = getattr(self, "_tp_lora_adapter", None)
        if adapter is not None:
            # The adapter owns TP/SP-specific communication so it can choose
            # fused async LoRA kernels without reimplementing layer dispatch here.
            return adapter(x)
        if self.sequence_parallel and self._group is not None and self._group.size() > 1:
            # SP path: fused async Function (backward overlaps AG/RS with matmul)
            return _ColumnLinearFwdBwd.apply(
                x, self.weight, self.bias, self._group, self.seq_dim,
                self.reuse_fwd_gather_for_wgrad,
            )
        # Non-SP or single-GPU: existing path (copy_to_tp_region for cross-attn KV)
        return self.forward_from_prepared_input(self._prepare_tp_input(x))

    def trim_full_output(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Trim a full gathered output tensor back to original features."""
        if self.packed_parts is None:
            return trim_padded_features(x, self.original_out_features, dim=dim)
        return trim_packed_padded_features(
            x,
            packed_parts=self.packed_parts,
            original_part_size=self.original_part_size,
            padded_part_size=self.padded_part_size,
            dim=dim,
        )

    def split_local_packed_output(self, x: torch.Tensor, dim: int = -1) -> tuple[torch.Tensor, ...]:
        """Split local packed-column output using padded per-part shard sizes."""
        if self.packed_parts is None:
            raise ValueError("split_local_packed_output requires packed_parts")
        dim = dim % x.ndim
        expected = self.packed_parts * self.local_part_size
        if x.size(dim) != expected:
            raise ValueError(
                f"local packed dim size {x.size(dim)} does not match expected {expected}"
            )
        return tuple(x.split(self.local_part_size, dim=dim))


class RowParallelLinear(nn.Module):
    """
    Row-parallel linear layer.

    sequence_parallel=True (default):
        Forward: local matmul + reduce-scatter along seq_dim.
                 Input (…, S, …, D_in/tp) → partial (…, S, …, D_out) → scatter (…, S/tp, …, D_out)
        Backward: all-gather gradient.

    sequence_parallel=False:
        Forward: local matmul + all-reduce.
                 Use when output stays replicated (e.g. no SP region after this layer).
        Backward: local input gradient (all-reduce is its own backward).

    seq_dim: which dimension holds the sequence tokens (default 0).
        seq_dim=0 → Megatron (S, B, D)
        seq_dim=1 → batch-first (B, S, D)  ← use this for Anima / HuggingFace models
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        bias:         bool = False,
        sequence_parallel: bool = True,
        seq_dim:      int  = 0,
    ):
        super().__init__()
        self.in_features       = in_features
        self.out_features      = out_features
        self.sequence_parallel = sequence_parallel
        self.seq_dim           = seq_dim
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None
        self._group: Optional[dist.ProcessGroup] = None

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    @classmethod
    def from_linear(
        cls,
        linear:       nn.Linear,
        rank:         int,
        world_size:   int,
        group:        dist.ProcessGroup,
        sequence_parallel: bool = True,
        seq_dim:      int  = 0,
        allow_padding: bool = False,
        padding_multiple: int = 1,
    ) -> "RowParallelLinear":
        """Create from an existing nn.Linear, slicing its weight rowwise."""
        if linear.in_features % world_size != 0 and not allow_padding:
            raise ValueError(
                f"in_features={linear.in_features} is not divisible by "
                f"tp_size={world_size}"
            )
        padded_in_features = (
            _padded_size_for_tp(linear.in_features, world_size, padding_multiple)
            if allow_padding else linear.in_features
        )
        in_shard = padded_in_features // world_size
        layer = cls(
            in_features       = in_shard,
            out_features      = linear.out_features,
            bias              = linear.bias is not None,
            sequence_parallel = sequence_parallel,
            seq_dim           = seq_dim,
        )
        layer.weight = nn.Parameter(
            _shard_rowwise(
                linear.weight.data,
                rank,
                world_size,
                allow_padding=allow_padding,
                padding_multiple=padding_multiple,
            )
        )
        if linear.bias is not None:
            layer.bias = nn.Parameter(linear.bias.data.clone())
        layer._group = group
        layer.original_in_features = linear.in_features
        layer.original_out_features = linear.out_features
        layer.padded_in_features = padded_in_features
        layer.padded_out_features = linear.out_features
        layer.allow_padding = allow_padding
        layer.padding_multiple = padding_multiple
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) < self.in_features:
            x = torch.nn.functional.pad(x, (0, self.in_features - x.size(-1)))
        # Fused async Function: matmul → reduce-scatter/all-reduce, bias added inside.
        # Backward launches async AG of grad_out to enable cross-layer pipelining.
        return _RowLinearFwdBwd.apply(
            x, self.weight, self.bias, self._group, self.seq_dim, self.sequence_parallel
        )
