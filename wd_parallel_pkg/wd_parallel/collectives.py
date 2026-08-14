"""
Communication primitives for TP+SP.

All collectives use dist.all_gather_into_tensor / dist.reduce_scatter_tensor
directly — no DTensor functional collectives. This ensures every collective
routes through whatever backend is registered (cuda_direct, gloo, NCCL).

Tensor layout convention (Megatron-style): sequence is dim 0.
  Sequence-parallel region: (S/tp, B, D)
  TP region (after column gather): (S, B, D_local)

The ``seq_dim`` parameter generalises the SP collectives beyond sequence-first
layout.  Specify which dimension holds the sequence tokens:

  seq_dim=0  (default) — Megatron / Lumina style: (S, B, D)
  seq_dim=1            — batch-first style:        (B, S, D)  ← Anima, HuggingFace
  seq_dim=2            — e.g. (B, heads, S, D) for some custom models

All public API functions and the autograd functions accept an optional
``seq_dim`` keyword that defaults to 0, so existing callers are unaffected.
Non-zero seq_dim transpose to dim-0 internally, run the collective, then
transpose back.  This incurs a contiguous copy but is unavoidable because
dist.all_gather_into_tensor / dist.reduce_scatter_tensor always scatter/gather
along the first dimension of the flat storage.

Autograd symmetry:
  _GatherFromSPRegion:   forward=all-gather,    backward=reduce-scatter
  _ReduceScatterToSPRegion: forward=reduce-scatter, backward=all-gather
  _CopyToTPRegion:       forward=identity,       backward=all-reduce

GlobalMemoryBuffer keeps one output tensor per buf_name and reallocates only
when shape/dtype/device changes.  This avoids torch.empty() fragmentation on
stable-shape workloads while bounding memory to one buffer per name, so
bucketed / variable-length training can't inflate the pool.  The seq_dim=0
path uses this pool directly; the non-zero seq_dim path bypasses it because
the return tensor is already a fresh allocation (post-transpose contiguous
copy).
"""

import time

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Optional comm timer (enable for profiling)
# ---------------------------------------------------------------------------

class CommTimer:
    """Accumulates wall time in collective ops. Enable with .enabled = True."""

    def __init__(self):
        self.enabled     = False
        self.fwd_ms      = 0.0
        self.bwd_ms      = 0.0
        self._in_backward = False

    def reset(self):
        self.fwd_ms = self.bwd_ms = 0.0
        self._in_backward = False

    def mark_backward(self):
        self._in_backward = True

    def _record(self, ms: float):
        if self._in_backward:
            self.bwd_ms += ms
        else:
            self.fwd_ms += ms

    @property
    def total_ms(self) -> float:
        return self.fwd_ms + self.bwd_ms


comm_timer = CommTimer()   # module-level singleton


# ---------------------------------------------------------------------------
# Pre-allocated buffer pool (fixes GAP 2)
# ---------------------------------------------------------------------------

class GlobalMemoryBuffer:
    """
    One buffer per name. Reuses when shape/dtype/device match, reallocates
    otherwise. Bounded at ``len(buf_names)`` regardless of how many unique
    shapes are seen — safe under bucketed/variable-length training.
    """

    def __init__(self):
        self._buffers: dict = {}

    def get(self, shape: tuple, dtype: torch.dtype,
            device: torch.device, name: str) -> torch.Tensor:
        buf = self._buffers.get(name)
        if (buf is None
                or tuple(buf.shape) != tuple(shape)
                or buf.dtype != dtype
                or buf.device != device):
            buf = torch.empty(shape, dtype=dtype, device=device)
            self._buffers[name] = buf
        return buf

    def clear(self):
        self._buffers.clear()


_memory_buffer = GlobalMemoryBuffer()


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------

def _maybe_cuda_sync(input_: torch.Tensor) -> None:
    if input_.is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()


def _maybe_time_collective(input_: torch.Tensor, fn):
    if not comm_timer.enabled:
        fn()
        return
    _maybe_cuda_sync(input_)
    t0 = time.perf_counter()
    fn()
    _maybe_cuda_sync(input_)
    comm_timer._record((time.perf_counter() - t0) * 1e3)


def _gather_along_first_dim(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    buf_name: str = "gather",
    use_buffer: bool = True,
) -> torch.Tensor:
    """All-gather along dim 0: (S/tp, B, D) → (S, B, D)."""
    world_size = group.size()
    if world_size == 1:
        return input_

    out_shape    = (input_.size(0) * world_size,) + input_.shape[1:]
    if use_buffer:
        output = _memory_buffer.get(out_shape, input_.dtype, input_.device, buf_name)
    else:
        output = torch.empty(out_shape, dtype=input_.dtype, device=input_.device)

    _maybe_time_collective(
        input_,
        lambda: dist.all_gather_into_tensor(output, input_.contiguous(), group=group),
    )

    return output


def _reduce_scatter_along_first_dim(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    buf_name: str = "reduce_scatter",
    use_buffer: bool = True,
) -> torch.Tensor:
    """Reduce-scatter along dim 0: (S, B, D) → (S/tp, B, D)."""
    world_size = group.size()
    if world_size == 1:
        return input_

    if input_.size(0) % world_size != 0:
        raise ValueError(
            f"dim-0 size {input_.size(0)} not divisible by world_size {world_size}"
        )
    out_shape = (input_.size(0) // world_size,) + input_.shape[1:]
    if use_buffer:
        output = _memory_buffer.get(out_shape, input_.dtype, input_.device, buf_name)
    else:
        output = torch.empty(out_shape, dtype=input_.dtype, device=input_.device)

    _maybe_time_collective(
        input_,
        lambda: dist.reduce_scatter_tensor(output, input_.contiguous(), group=group),
    )

    return output


def _split_along_first_dim(input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Local split on dim 0: returns this rank's chunk. No communication."""
    world_size = group.size()
    if world_size == 1:
        return input_
    if input_.size(0) % world_size != 0:
        raise ValueError(
            f"dim-0 size {input_.size(0)} not divisible by world_size {world_size}"
        )
    rank        = dist.get_rank(group=group)
    chunk       = input_.size(0) // world_size
    return input_[rank * chunk : (rank + 1) * chunk].contiguous()


# ---------------------------------------------------------------------------
# Generalised dim-aware wrappers (seq_dim != 0 support)
# ---------------------------------------------------------------------------


def _apply_with_seq_dim_first(
    input_: torch.Tensor,
    seq_dim: int,
    op,
) -> torch.Tensor:
    if seq_dim == 0:
        return op(input_)
    x = input_.transpose(0, seq_dim).contiguous()
    y = op(x)
    return y.transpose(0, seq_dim).contiguous()

def _gather_along_dim(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    seq_dim: int = 0,
    buf_name: str = "gather",
) -> torch.Tensor:
    """All-gather along an arbitrary sequence dimension.

    Always returns a tensor whose storage is private to the caller:
      seq_dim=0: pool-backed output is cloned before return.
      seq_dim!=0: _apply_with_seq_dim_first produces a fresh tensor via
                  the post-transpose .contiguous(); no clone needed.
    """
    if seq_dim == 0:
        return _gather_along_first_dim(input_, group, buf_name, use_buffer=True).clone()
    return _apply_with_seq_dim_first(
        input_,
        seq_dim,
        lambda x: _gather_along_first_dim(x, group, buf_name, use_buffer=False),
    )


def _reduce_scatter_along_dim(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    seq_dim: int = 0,
    buf_name: str = "reduce_scatter",
) -> torch.Tensor:
    """Reduce-scatter along an arbitrary sequence dimension.

    Mirrors _gather_along_dim: seq_dim=0 clones the pool-backed result;
    seq_dim!=0 returns the fresh tensor from _apply_with_seq_dim_first.
    """
    if seq_dim == 0:
        return _reduce_scatter_along_first_dim(input_, group, buf_name, use_buffer=True).clone()
    return _apply_with_seq_dim_first(
        input_,
        seq_dim,
        lambda x: _reduce_scatter_along_first_dim(x, group, buf_name, use_buffer=False),
    )


def _split_along_dim(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    seq_dim: int = 0,
) -> torch.Tensor:
    """Local split along an arbitrary dimension — no communication.

    seq_dim=0 is a direct slice (fast path).  Other dims transpose first.
    """
    return _apply_with_seq_dim_first(
        input_, seq_dim, lambda x: _split_along_first_dim(x, group)
    )


def pad_to_world_size(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    seq_dim: int = 0,
) -> tuple[torch.Tensor, int]:
    """Pad ``seq_dim`` to a multiple of the group size and return (tensor, pad)."""
    world_size = group.size()
    seq_dim = seq_dim % input_.ndim
    size = input_.size(seq_dim)
    remainder = size % world_size
    pad = 0 if remainder == 0 else world_size - remainder
    if pad == 0:
        return input_, 0
    pad_shape = list(input_.shape)
    pad_shape[seq_dim] = pad
    padding = input_.new_zeros(pad_shape)
    return torch.cat([input_, padding], dim=seq_dim).contiguous(), pad


def split_along_dim_with_padding(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    seq_dim: int = 0,
) -> tuple[torch.Tensor, int, int]:
    """Pad then local-split along ``seq_dim``.

    Returns:
        local_shard, original_size, pad
    """
    original_size = input_.size(seq_dim)
    padded, pad = pad_to_world_size(input_, group, seq_dim)
    return _split_along_dim(padded, group, seq_dim), original_size, pad


def gather_from_sp_region_and_trim(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    seq_dim: int = 0,
    *,
    original_size: int | None = None,
    pad: int = 0,
) -> torch.Tensor:
    """All-gather an SP shard and remove padding on ``seq_dim``."""
    gathered = gather_from_sp_region(input_, group, seq_dim)
    seq_dim = seq_dim % gathered.ndim
    if original_size is None:
        if pad <= 0:
            return gathered
        original_size = gathered.size(seq_dim) - pad
    index = [slice(None)] * gathered.ndim
    index[seq_dim] = slice(0, original_size)
    return gathered[tuple(index)].contiguous()


# ---------------------------------------------------------------------------
# Differentiable autograd Functions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# First-NaN tracker (one report per training run, rank 0 only)
# ---------------------------------------------------------------------------

_nan_report_count = 0   # how many NaN events we've reported so far
_NAN_REPORT_LIMIT = 3   # print at most this many distinct NaN events
_nan_diagnostics_enabled = False


def reset_nan_diagnostics() -> None:
    """Reset the first-NaN counter. Call at the start of each training run."""
    global _nan_report_count
    _nan_report_count = 0


def set_nan_diagnostics_enabled(enabled: bool) -> None:
    """Enable/disable expensive NaN scanning diagnostics on collective outputs."""
    global _nan_diagnostics_enabled
    _nan_diagnostics_enabled = bool(enabled)


def _maybe_report_nan(tag: str, inp: torch.Tensor, out: torch.Tensor) -> None:
    """Print a one-line diagnostic when a collective receives or produces NaN.

    Prints only on rank 0 (avoids duplicate output on all TP ranks).
    Stops after _NAN_REPORT_LIMIT events so the log isn't flooded.
    """
    global _nan_report_count
    if not _nan_diagnostics_enabled:
        return
    if _nan_report_count >= _NAN_REPORT_LIMIT:
        return
    inp_nan = not torch.isfinite(inp).all()
    out_nan = not torch.isfinite(out).all()
    if not inp_nan and not out_nan:
        return
    try:
        rank = dist.get_rank()
    except Exception:
        rank = 0
    if rank != 0:
        return
    _nan_report_count += 1
    try:
        from tqdm import tqdm
        write = tqdm.write
    except ImportError:
        import builtins
        write = builtins.print
    source = "INPUT" if inp_nan else "COLLECTIVE"
    write(
        f"[SP NaN #{_nan_report_count}] {tag}: {source} has NaN/Inf  "
        f"inp_nan={inp_nan} out_nan={out_nan}  "
        f"inp_dtype={inp.dtype} out_dtype={out.dtype}  "
        f"inp_shape={tuple(inp.shape)}"
    )


class _GatherFromSPRegion(torch.autograd.Function):
    """
    Forward:  all-gather along seq_dim    (SP region → TP region)
    Backward: reduce-scatter along seq_dim (TP grad → SP grad)

    seq_dim selects which tensor dimension holds the sequence tokens:
      seq_dim=0  Megatron-style (S, B, D)  — fast buffered path
      seq_dim=1  batch-first   (B, S, D)  — transpose path
    """
    @staticmethod
    def forward(ctx, input_, group, seq_dim):
        ctx.group      = group
        ctx.seq_dim    = seq_dim
        ctx.input_dtype = input_.dtype
        # _gather_along_dim returns a caller-private tensor (pool-backed
        # results are cloned internally for seq_dim=0; non-zero seq_dim
        # produces a fresh tensor from _apply_with_seq_dim_first).
        return _gather_along_dim(input_, group, seq_dim, buf_name="fwd_gather")

    @staticmethod
    def backward(ctx, grad_output):
        # cuda_direct only supports the same dtypes as the forward input.
        # During gradient-checkpointed backward passes, PyTorch may produce
        # float32 grad tensors even when the forward ran in bfloat16.  Cast
        # back to the forward input dtype so the collective never sees a
        # mismatched dtype.
        if grad_output.dtype != ctx.input_dtype:
            grad_output = grad_output.to(ctx.input_dtype)
        result = _reduce_scatter_along_dim(
            grad_output, ctx.group, ctx.seq_dim, buf_name="bwd_reduce_scatter"
        )
        _maybe_report_nan("GatherBwd(reduce_scatter)", grad_output, result)
        return (result, None, None)   # group, seq_dim need no gradient


class _ReduceScatterToSPRegion(torch.autograd.Function):
    """
    Forward:  reduce-scatter along seq_dim (TP region → SP region)
              Fuses TP all-reduce + SP scatter into one op — half the bandwidth vs all-reduce.
    Backward: all-gather along seq_dim     (SP grad → TP grad)
    """
    @staticmethod
    def forward(ctx, input_, group, seq_dim):
        ctx.group      = group
        ctx.seq_dim    = seq_dim
        ctx.input_dtype = input_.dtype
        return _reduce_scatter_along_dim(input_, group, seq_dim, buf_name="fwd_reduce_scatter")

    @staticmethod
    def backward(ctx, grad_output):
        # Same dtype-safety cast as _GatherFromSPRegion.backward.
        if grad_output.dtype != ctx.input_dtype:
            grad_output = grad_output.to(ctx.input_dtype)
        result = _gather_along_dim(
            grad_output, ctx.group, ctx.seq_dim, buf_name="bwd_gather"
        )
        _maybe_report_nan("ReduceScatterBwd(all_gather)", grad_output, result)
        return (result, None, None)   # group, seq_dim need no gradient


class _CopyToTPRegion(torch.autograd.Function):
    """
    Forward:  identity (replicated input, no comm)
    Backward: all-reduce (sum gradients across TP ranks)
    Used for cross-attention K/V projections whose input is replicated context.
    seq_dim is unused here (no collective along sequence in forward).

    NOTE: forward must return a clone, NOT input_ directly.
    Returning the same tensor in PyTorch 2.7+ causes the input to be promoted
    to non-leaf (aliased-output detection), silently breaking the backward graph.
    """
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        ctx.input_dtype = input_.dtype
        return input_.clone()  # clone avoids aliasing; backward math is unchanged

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output.dtype != ctx.input_dtype:
            grad_output = grad_output.to(ctx.input_dtype)
        grad = grad_output.contiguous()
        dist.all_reduce(grad, group=ctx.group)
        _maybe_report_nan("CopyToTPBwd(all_reduce)", grad_output, grad)
        return grad, None


# ---------------------------------------------------------------------------
# Async helpers for fused linear backward overlap
# ---------------------------------------------------------------------------

def _async_all_gather_seq_dim(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    seq_dim: int,
):
    """Non-blocking all-gather along seq_dim.

    Returns (out_buf, handle, needs_back_transpose).
    After handle.wait(), caller must do out_buf.transpose(0, seq_dim).contiguous()
    when needs_back_transpose is True to restore the original layout.

    Fresh (non-pooled) buffers are always used so concurrent async ops don't
    alias each other's storage.

    Requires CUDA_DEVICE_MAX_CONNECTIONS=1 for compute-comm overlap to work.
    """
    world_size = group.size()
    if seq_dim == 0:
        x = input_.contiguous()
        out_shape = (x.size(0) * world_size,) + x.shape[1:]
        buf = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        handle = dist.all_gather_into_tensor(buf, x, group=group, async_op=True)
        return buf, handle, False
    else:
        x = input_.transpose(0, seq_dim).contiguous()
        out_shape = (x.size(0) * world_size,) + x.shape[1:]
        buf = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        handle = dist.all_gather_into_tensor(buf, x, group=group, async_op=True)
        return buf, handle, True


def _async_reduce_scatter_seq_dim(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    seq_dim: int,
):
    """Non-blocking reduce-scatter along seq_dim.

    Returns (out_buf, handle, needs_back_transpose).
    After handle.wait(), caller must do out_buf.transpose(0, seq_dim).contiguous()
    when needs_back_transpose is True to restore the original layout.
    """
    world_size = group.size()
    if seq_dim == 0:
        x = input_.contiguous()
        out_shape = (x.size(0) // world_size,) + x.shape[1:]
        buf = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        handle = dist.reduce_scatter_tensor(buf, x, group=group, async_op=True)
        return buf, handle, False
    else:
        x = input_.transpose(0, seq_dim).contiguous()
        out_shape = (x.size(0) // world_size,) + x.shape[1:]
        buf = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        handle = dist.reduce_scatter_tensor(buf, x, group=group, async_op=True)
        return buf, handle, True


class _ColumnLinearFwdBwd(torch.autograd.Function):
    """Fused column-parallel linear with async backward overlap.

    Forward  (sync):  all-gather input along seq_dim → matmul.
    Backward (async, requires CUDA_DEVICE_MAX_CONNECTIONS=1):
      1. async AG(saved input shard)              -- enqueue, return handle
      2. grad_input = grad_out @ weight           -- overlaps with AG
      3. async RS(grad_input → SP region)         -- enqueue, return handle
      4. handle_ag.wait()
      5. wgrad = grad_out_2d.T @ gathered_2d      -- overlaps with RS
      6. handle_rs.wait()
      -> return sharded grad_input, wgrad, grad_bias

    When weight.requires_grad is False (frozen base in LoRA training):
      steps 1/4/5 are skipped — only 2/3/6.

    save_gathered: when True (and weight.requires_grad and group.size()>1),
      forward saves the already-gathered full-sequence activation instead of
      the SP shard, so backward reuses it for wgrad instead of re-running the
      AG from step 1. Removes one collective per call at the cost of holding
      the full-sequence activation across the backward (bounded — callers
      should only enable this under gradient checkpointing, where that
      activation is about to be recomputed/freed anyway).
    """

    @staticmethod
    def forward(ctx, input_, weight, bias, group, seq_dim, save_gathered=False):
        ctx.group            = group
        ctx.seq_dim          = seq_dim
        ctx.use_bias         = bias is not None
        ctx.input_dtype      = input_.dtype
        ctx.weight_needs_grad = weight.requires_grad

        if group is not None and group.size() > 1:
            gathered = _gather_along_dim(input_, group, seq_dim, buf_name="col_fwd_gather")
        else:
            gathered = input_

        reused_gather = (
            bool(save_gathered)
            and ctx.weight_needs_grad
            and group is not None
            and group.size() > 1
        )
        ctx.reused_gather = reused_gather
        ctx.save_for_backward(gathered if reused_gather else input_, weight)

        return torch.nn.functional.linear(gathered, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        saved, weight = ctx.saved_tensors
        group   = ctx.group
        seq_dim = ctx.seq_dim
        reused  = ctx.reused_gather

        # Cast grad to weight dtype — weight.dtype is authoritative for matmul ops.
        # ctx.input_dtype may be fp32 when AMP autocast upcasts activations, while
        # the weight stays in bf16; casting to input_dtype then matmul-ing with bf16
        # weight would crash.
        if grad_output.dtype != weight.dtype:
            grad_output = grad_output.to(weight.dtype)

        do_async = group is not None and group.size() > 1

        # Step 1: async AG of saved input shard (for wgrad) — launch before grad_input matmul
        # CUDA_DEVICE_MAX_CONNECTIONS=1 ensures AG is enqueued before the matmul kernel below.
        # Skipped when `reused` — the forward already gathered this exact tensor.
        if do_async and ctx.weight_needs_grad and not reused:
            ag_buf, ag_handle, ag_transpose = _async_all_gather_seq_dim(saved, group, seq_dim)
        else:
            ag_buf = ag_handle = ag_transpose = None

        # Step 2: grad_input (full sequence) — overlaps with AG when weight_needs_grad=True
        grad_input_full = grad_output.matmul(weight)

        # Step 3: async RS of grad_input back to SP region
        if do_async:
            rs_buf, rs_handle, rs_transpose = _async_reduce_scatter_seq_dim(
                grad_input_full, group, seq_dim
            )

        # Step 4+5: wgrad. Three cases: reused forward gather (no AG needed),
        # awaited async AG, or single-GPU (saved tensor is already the full input).
        if reused:
            gathered = saved
            if gathered.dtype != weight.dtype:
                gathered = gathered.to(weight.dtype)
            gathered_2d = gathered.reshape(-1, weight.shape[1])
            grad_out_2d = grad_output.reshape(-1, grad_output.shape[-1])
            grad_weight = grad_out_2d.t().matmul(gathered_2d)
        elif ag_handle is not None:
            ag_handle.wait()
            gathered = ag_buf.transpose(0, seq_dim).contiguous() if ag_transpose else ag_buf
            if gathered.dtype != weight.dtype:
                gathered = gathered.to(weight.dtype)
            gathered_2d = gathered.reshape(-1, weight.shape[1])
            grad_out_2d = grad_output.reshape(-1, grad_output.shape[-1])
            grad_weight = grad_out_2d.t().matmul(gathered_2d)
        elif ctx.weight_needs_grad:
            # Single-GPU: full input is already on this rank
            inp = saved.to(weight.dtype) if saved.dtype != weight.dtype else saved
            inp_2d      = inp.reshape(-1, weight.shape[1])
            grad_out_2d = grad_output.reshape(-1, grad_output.shape[-1])
            grad_weight = grad_out_2d.t().matmul(inp_2d)
        else:
            grad_weight = None

        # Step 6: wait for RS, restore layout
        if do_async:
            rs_handle.wait()
            grad_input = rs_buf.transpose(0, seq_dim).contiguous() if rs_transpose else rs_buf
        else:
            grad_input = grad_input_full

        grad_bias = (
            grad_output.sum(list(range(grad_output.ndim - 1))) if ctx.use_bias else None
        )

        # group, seq_dim, save_gathered: no grad
        return (grad_input, grad_weight, grad_bias, None, None, None)


class _RowLinearFwdBwd(torch.autograd.Function):
    """Fused row-parallel linear with async backward AG.

    Forward  (sync):  matmul with feature-sharded input → reduce-scatter to SP region.
    Backward:
      1. async AG(grad_out SP shard)              -- gather gradient to full sequence
      2. handle.wait()                            -- both grad_input and wgrad need full grad_out
      3. grad_input = grad_out_full @ weight      -- feature-sharded, no collective
      4. wgrad      = grad_out_2d.T @ input_2d   -- local, no collective
      -> return grad_input, wgrad, grad_bias

    No inner overlap in RowParallel backward (unlike Column), but async AG enables
    cross-layer pipelining when CUDA_DEVICE_MAX_CONNECTIONS=1 is set.
    """

    @staticmethod
    def forward(ctx, input_, weight, bias, group, seq_dim, sequence_parallel):
        ctx.save_for_backward(input_, weight)
        ctx.group             = group
        ctx.seq_dim           = seq_dim
        ctx.sequence_parallel = sequence_parallel
        ctx.use_bias          = bias is not None
        ctx.input_dtype       = input_.dtype
        ctx.weight_needs_grad = weight.requires_grad

        # Bias added AFTER collective to avoid tp_size-times accumulation
        out = torch.nn.functional.linear(input_, weight, None)

        if group is not None and group.size() > 1:
            if sequence_parallel:
                out = _reduce_scatter_along_dim(out, group, seq_dim, buf_name="row_fwd_rs")
            else:
                out = out.contiguous()
                dist.all_reduce(out, group=group)

        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        group   = ctx.group
        seq_dim = ctx.seq_dim
        sp      = ctx.sequence_parallel

        # Cast to weight dtype — same AMP reason as _ColumnLinearFwdBwd.
        if grad_output.dtype != weight.dtype:
            grad_output = grad_output.to(weight.dtype)

        do_async = group is not None and group.size() > 1

        # Step 1+2: async AG of SP-sharded grad_out to full sequence
        if do_async and sp:
            ag_buf, ag_handle, ag_transpose = _async_all_gather_seq_dim(grad_output, group, seq_dim)
            ag_handle.wait()
            grad_out_full = ag_buf.transpose(0, seq_dim).contiguous() if ag_transpose else ag_buf
        else:
            grad_out_full = grad_output

        # Step 3: grad_input — feature-sharded (matches row-parallel weight shard)
        grad_input = grad_out_full.matmul(weight)

        # Step 4: wgrad — fully local, no collective needed
        if ctx.weight_needs_grad:
            inp = input_.to(weight.dtype) if input_.dtype != weight.dtype else input_
            inp_2d      = inp.reshape(-1, weight.shape[1])
            grad_out_2d = grad_out_full.reshape(-1, grad_out_full.shape[-1])
            grad_weight = grad_out_2d.t().matmul(inp_2d)
        else:
            grad_weight = None

        grad_bias = (
            grad_output.sum(list(range(grad_output.ndim - 1))) if ctx.use_bias else None
        )

        return (grad_input, grad_weight, grad_bias, None, None, None)  # group, seq_dim, sp: no grad


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def gather_from_sp_region(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    seq_dim: int = 0,
) -> torch.Tensor:
    """All-gather sequence from SP region. Differentiable.

    Args:
        x:       Input tensor. The sequence dimension is ``seq_dim``.
        group:   TP process group.
        seq_dim: Dimension that holds the sequence tokens.
                 0 (default) → Megatron (S, B, D); 1 → batch-first (B, S, D).
    """
    return _GatherFromSPRegion.apply(x, group, seq_dim)


def reduce_scatter_to_sp_region(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    seq_dim: int = 0,
) -> torch.Tensor:
    """Reduce-scatter to SP region. Differentiable. Replaces TP all-reduce.

    Args:
        x:       Input tensor. The sequence dimension is ``seq_dim``.
        group:   TP process group.
        seq_dim: Dimension that holds the sequence tokens.
                 0 (default) → Megatron (S, B, D); 1 → batch-first (B, S, D).
    """
    return _ReduceScatterToSPRegion.apply(x, group, seq_dim)


def copy_to_tp_region(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Identity forward, all-reduce backward. For replicated inputs (e.g. cross-attn K/V)."""
    return _CopyToTPRegion.apply(x, group)


def copy_to_tp_region_no_input_grad(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Identity-like forward that intentionally stops input gradients.

    Use only when the replicated input is known to be frozen and no gradient
    should flow back into the source branch. This avoids the backward
    all-reduce that `copy_to_tp_region()` performs.
    """
    del group
    return x.detach().clone()
