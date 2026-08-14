# Anima full finetune training script — TP (Tensor Parallel) + SP (Sequence Parallel)
#
# TPSPTrainerMixin holds all TP/SP hook overrides and is combined with either
# AnimaTrainer (anima_train.py, standard optimizers) or MuonAnimaTrainer
# (anima_train_muon.py, Muon+AdamW hybrid) to produce AnimaTrainerTPSP /
# MuonAnimaTrainerTPSP. __main__ dispatches between them based on --use_muon.
# Only the hook methods that need TP/SP awareness are overridden; the full
# training loop is inherited from whichever base class is selected.
#
# Muon under TP uses muon_optimizer.TPMuonWithAuxAdam (see library/muon_optimizer.py),
# which all-gathers each TP-sharded weight's momentum update, runs Newton-Schulz
# orthogonalization on the reconstructed full matrix (matching the DDP/single-GPU
# math exactly), then re-shards the result — NOT the upstream MuonWithAuxAdam,
# whose round-robin all_gather assumes replicated params and would corrupt
# TP-sharded weights.
#
# Architecture mirrors anima_train_network_tensor_sequence_parallel.py:
#   - TP init happens in __main__ before trainer creation (not inside a hook)
#   - Accelerator state is spoofed so all TP ranks enter save/load hooks
#   - Batch is broadcast from rank 0 so all TP ranks see the same data
#   - QKV projections are fused internally, then wd_parallel shards them
#   - Attention head counts are fixed after sharding
#   - save_utils.install_parallel_state_wrappers is called for rank-aware saves
#
# Launch with torchrun:
#   torchrun --nproc_per_node=2 anima_train_tensor_sequence_parallel.py --tp_degree 2 [other args]
#   torchrun --nproc_per_node=2 anima_train_tensor_sequence_parallel.py --tp_degree 2 --use_muon [other args]
#
# tp_degree=1 falls back to plain AnimaTrainer/MuonAnimaTrainer (no TP init, single GPU).

import argparse
import os
import sys
import types
from typing import Union

import torch
import torch.distributed as dist

from library.device_utils import init_ipex
init_ipex()

from library import train_util, save_utils, huggingface_util
from library.utils import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

try:
    import wd_parallel as wdp
    _WDP_AVAILABLE = True
except ImportError:
    logger.warning("wd_parallel not found — TP disabled. Run: pip install -r requirements.txt")
    _WDP_AVAILABLE = False

from anima_train import AnimaTrainer
from anima_train_muon import MuonAnimaTrainer, setup_parser as _muon_setup_parser
from library import muon_optimizer


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _ceil_to_multiple(size: int, multiple: int) -> int:
    return ((size + multiple - 1) // multiple) * multiple


def _infer_anima_tp_padding_geometry(dit: torch.nn.Module, tp_size: int) -> dict:
    """Infer head-aligned padding geometry for Anima TP.

    Padding is internal and always enabled for this trainer. Divisible TP degrees
    naturally get padded_width == model_channels, so the old tp=2/4/8 behavior is
    unchanged for the 2048/16 model.
    """
    model_channels = int(getattr(dit, "model_channels"))
    num_heads = int(getattr(dit, "num_heads"))
    if model_channels % num_heads != 0:
        raise ValueError(
            f"model_channels={model_channels} must be divisible by num_heads={num_heads}"
        )
    head_dim = model_channels // num_heads
    padded_width = _ceil_to_multiple(model_channels, tp_size * head_dim)
    local_width = padded_width // tp_size
    if local_width % head_dim != 0:
        raise ValueError(
            f"invalid TP padding geometry: local_width={local_width} is not "
            f"divisible by head_dim={head_dim}"
        )
    return {
        "model_channels": model_channels,
        "num_heads":       num_heads,
        "head_dim":        head_dim,
        "tp_size":         tp_size,
        "padded_width":    padded_width,
        "local_width":     local_width,
        "local_heads":     local_width // head_dim,
        "padding_added":   padded_width - model_channels,
    }


def _fixup_attention_heads_for_tp(model: torch.nn.Module) -> int:
    """After apply_parallelism(), set each Attention module to its local heads.

    ColumnParallelLinear shards the OUTPUT dimension (features), so q/k/v proj
    output local features per rank. Each module uses its own head_dim so that
    modules with different head sizes (e.g. main DiT vs LLM adapter) are all
    handled correctly.

    Returns the number of Attention modules updated.
    """
    from library.anima_models import Attention, LLMAdapterAttention
    from wd_parallel.layers import ColumnParallelLinear

    updated = 0
    for module in model.modules():
        proj = None
        if hasattr(module, "q_proj") and isinstance(module.q_proj, ColumnParallelLinear):
            proj = module.q_proj
            local_width = int(proj.out_features)
        elif hasattr(module, "qkv_proj") and isinstance(module.qkv_proj, ColumnParallelLinear):
            proj = module.qkv_proj
            local_width = int(getattr(proj, "local_part_size", proj.out_features // 3))
        else:
            continue
        if isinstance(module, (Attention, LLMAdapterAttention)):
            mod_head_dim = int(module.head_dim)
            if local_width % mod_head_dim != 0:
                raise ValueError(
                    f"{type(module).__name__}: local q width={local_width} "
                    f"is not divisible by head_dim={mod_head_dim}"
                )
            module.n_heads = local_width // mod_head_dim
            updated += 1
    return updated


def _mark_replicated_context_layers_no_input_grad(
    model: torch.nn.Module,
    *,
    text_encoder_frozen: bool,
) -> int:
    """Mark replicated-input TP column layers whose input grad can be skipped.

    Safe only when the text encoder / conditioning source is frozen. Targets
    cross-attention K/V projections that consume replicated context.
    """
    if not text_encoder_frozen:
        return 0

    marked = 0
    suffixes = (
        ".cross_attn.kv_proj",
        ".cross_attn.k_proj",
        ".cross_attn.v_proj",
    )
    for name, module in model.named_modules():
        if not any(name.endswith(suffix) for suffix in suffixes):
            continue
        if getattr(module, "sequence_parallel", True):
            continue
        if hasattr(module, "skip_input_grad"):
            module.skip_input_grad = True
            marked += 1
    return marked


# ---------------------------------------------------------------------------
# NaN diagnostics for FFT TP/SP layers
# ---------------------------------------------------------------------------

def _register_tp_nan_hooks(dit: torch.nn.Module) -> int:
    """Register backward NaN diagnostic hooks on the first ColPar and RowPar layers.

    Mirrors the diagnostics in lora_anima.ColumnParallelLoRAModule /
    RowParallelLoRAModule but hooks into the raw layer outputs instead of
    LoRA intermediates, since FFT has no LoRA path.

    Hook legend (same as LoRA diag, so logs are comparable):
      ColPar A  — output grad of the column-parallel matmul
      ColPar C  — input grad flowing back into the SP gather backward
      RowPar A  — output grad of the row-parallel matmul (post-scatter)
      RowPar C  — pre-scatter combined grad (full-sequence, before reduce-scatter)
    """
    from wd_parallel.layers import ColumnParallelLinear, RowParallelLinear

    _col_done = [False]
    _row_done = [False]
    n_hooks   = 0

    for mod_name, module in dit.named_modules():
        if isinstance(module, ColumnParallelLinear) and not _col_done[0]:
            _col_done[0] = True
            _n = mod_name
            _registered = [False]

            def _col_forward_hook(mod, inp, out, _n=_n, _reg=_registered):
                if _reg[0]:
                    return
                _reg[0] = True

                def _hook_output(grad):
                    from tqdm import tqdm
                    status = "NaN" if not torch.isfinite(grad).all() else "ok"
                    tqdm.write(
                        f"[NaN DIAG ColPar] A(output grad): {status}  {_n}"
                        f"  nan={torch.isnan(grad).sum().item()}  shape={tuple(grad.shape)}"
                    )
                    return grad

                x = inp[0] if isinstance(inp, (tuple, list)) else inp

                def _hook_input(grad):
                    from tqdm import tqdm
                    status = "NaN" if not torch.isfinite(grad).all() else "ok"
                    tqdm.write(
                        f"[NaN DIAG ColPar] C(input grad, after SP gather bwd): {status}"
                        f"  nan={torch.isnan(grad).sum().item()}"
                    )
                    return grad

                if out.requires_grad:
                    out.register_hook(_hook_output)
                if x is not None and x.requires_grad:
                    x.register_hook(_hook_input)

            module.register_forward_hook(_col_forward_hook)
            n_hooks += 1

        elif isinstance(module, RowParallelLinear) and not _row_done[0]:
            _row_done[0] = True
            _n = mod_name
            _registered = [False]

            def _row_forward_hook(mod, inp, out, _n=_n, _reg=_registered):
                if _reg[0]:
                    return
                _reg[0] = True

                def _hook_output(grad):
                    from tqdm import tqdm
                    status = "NaN" if not torch.isfinite(grad).all() else "ok"
                    tqdm.write(
                        f"[NaN DIAG RowPar] A(output grad): {status}  {_n}"
                        f"  nan={torch.isnan(grad).sum().item()}  shape={tuple(grad.shape)}"
                    )
                    return grad

                x = inp[0] if isinstance(inp, (tuple, list)) else inp

                def _hook_input(grad):
                    from tqdm import tqdm
                    status = "NaN" if not torch.isfinite(grad).all() else "ok"
                    tqdm.write(
                        f"[NaN DIAG RowPar] C(input grad, pre-scatter): {status}"
                        f"  nan={torch.isnan(grad).sum().item()}"
                    )
                    return grad

                if out.requires_grad:
                    out.register_hook(_hook_output)
                if x is not None and x.requires_grad:
                    x.register_hook(_hook_input)

            module.register_forward_hook(_row_forward_hook)
            n_hooks += 1

        if _col_done[0] and _row_done[0]:
            break

    return n_hooks


# ---------------------------------------------------------------------------
# QKV fusion / unfusion
# ---------------------------------------------------------------------------

def fuse_qkv_for_tp(model: torch.nn.Module, *, include_llm_adapter: bool = True) -> int:
    """Fuse separate Q/K/V projections into combined linear layers, in-place.

    Idempotent: already-fused modules are skipped.
    Self-attention:  q_proj + k_proj + v_proj → qkv_proj  (3*inner_dim output)
    Cross-attention: k_proj + v_proj           → kv_proj   (2*inner_dim output)
                     q_proj stays separate     (different input: x vs context)
    LLMAdapterAttention is also fused when include_llm_adapter=True.

    TP benefit: cuts self-attn from 3 ColumnParallel all-gathers to 1,
    cross-attn from 3 to 2.  With 28 blocks: 84 → 42 all-gathers for QKV.
    """
    import types
    from einops import rearrange
    from library.anima_models import Attention, LLMAdapterAttention, apply_rotary_pos_emb
    import torch.nn.functional as F

    def _fused_self_attn_compute_qkv(self, x, context=None, rope_emb=None):
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        n_h = q.shape[-1] // self.head_dim
        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=n_h, d=self.head_dim),
            (q, k, v),
        )
        q, k, v = self.q_norm(q), self.k_norm(k), self.v_norm(v)
        if rope_emb is not None:
            q = apply_rotary_pos_emb(q, rope_emb, tensor_format=self.qkv_format, fused=False)
            k = apply_rotary_pos_emb(k, rope_emb, tensor_format=self.qkv_format, fused=False)
        return q, k, v

    def _fused_cross_attn_compute_qkv(self, x, context=None, rope_emb=None):
        q = self.q_proj(x)
        ctx = x if context is None else context
        k, v = self.kv_proj(ctx).chunk(2, dim=-1)
        n_h = q.shape[-1] // self.head_dim
        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=n_h, d=self.head_dim),
            (q, k, v),
        )
        return self.q_norm(q), self.k_norm(k), self.v_norm(v)

    def _fused_adapter_forward(self, x, mask=None, context=None, position_embeddings=None, position_embeddings_context=None):
        from library.anima_models import _adapter_apply_rotary_pos_emb
        context = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        context_shape = context.shape[:-1]
        kv_shape = (*context_shape, self.n_heads, self.head_dim)

        if hasattr(self, "qkv_proj"):
            q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        else:
            q = self.q_proj(x)
            k, v = self.kv_proj(context).chunk(2, dim=-1)

        query_states = self.q_norm(q.view(q_shape)).transpose(1, 2)
        key_states   = self.k_norm(k.view(kv_shape)).transpose(1, 2)
        value_states = v.view(kv_shape).transpose(1, 2)

        if position_embeddings is not None:
            assert position_embeddings_context is not None
            cos, sin = position_embeddings
            query_states = _adapter_apply_rotary_pos_emb(query_states, cos, sin)
            cos, sin = position_embeddings_context
            key_states   = _adapter_apply_rotary_pos_emb(key_states, cos, sin)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=mask)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)

    def _fuse_main_attention(attn: Attention) -> bool:
        if hasattr(attn, "qkv_proj") or hasattr(attn, "kv_proj"):
            return False
        if attn.is_selfattn:
            fused_w = torch.cat([attn.q_proj.weight.data, attn.k_proj.weight.data, attn.v_proj.weight.data], dim=0)
            attn.qkv_proj = torch.nn.Linear(
                attn.q_proj.in_features, fused_w.shape[0], bias=False,
                device=attn.q_proj.weight.device, dtype=attn.q_proj.weight.dtype,
            )
            attn.qkv_proj.weight = torch.nn.Parameter(fused_w)
            del attn.q_proj, attn.k_proj, attn.v_proj
            attn.compute_qkv = types.MethodType(_fused_self_attn_compute_qkv, attn)
        else:
            fused_w = torch.cat([attn.k_proj.weight.data, attn.v_proj.weight.data], dim=0)
            attn.kv_proj = torch.nn.Linear(
                attn.k_proj.in_features, fused_w.shape[0], bias=False,
                device=attn.k_proj.weight.device, dtype=attn.k_proj.weight.dtype,
            )
            attn.kv_proj.weight = torch.nn.Parameter(fused_w)
            del attn.k_proj, attn.v_proj
            attn.compute_qkv = types.MethodType(_fused_cross_attn_compute_qkv, attn)
        return True

    def _fuse_adapter_attention(attn: LLMAdapterAttention, *, is_self_attn: bool) -> bool:
        if hasattr(attn, "qkv_proj") or hasattr(attn, "kv_proj"):
            return False
        if is_self_attn:
            fused_w = torch.cat([attn.q_proj.weight.data, attn.k_proj.weight.data, attn.v_proj.weight.data], dim=0)
            attn.qkv_proj = torch.nn.Linear(
                attn.q_proj.in_features, fused_w.shape[0], bias=False,
                device=attn.q_proj.weight.device, dtype=attn.q_proj.weight.dtype,
            )
            attn.qkv_proj.weight = torch.nn.Parameter(fused_w)
            del attn.q_proj, attn.k_proj, attn.v_proj
        else:
            fused_w = torch.cat([attn.k_proj.weight.data, attn.v_proj.weight.data], dim=0)
            attn.kv_proj = torch.nn.Linear(
                attn.k_proj.in_features, fused_w.shape[0], bias=False,
                device=attn.k_proj.weight.device, dtype=attn.k_proj.weight.dtype,
            )
            attn.kv_proj.weight = torch.nn.Parameter(fused_w)
            del attn.k_proj, attn.v_proj
        attn.forward = types.MethodType(_fused_adapter_forward, attn)
        return True

    fused_count = 0
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            fused_count += int(_fuse_main_attention(module))
        elif include_llm_adapter and isinstance(module, LLMAdapterAttention):
            fused_count += int(_fuse_adapter_attention(module, is_self_attn=name.endswith(".self_attn")))
    logger.info(f"QKV fusion: {fused_count} attention modules fused.")
    return fused_count


def unfuse_qkv_from_tp(model: torch.nn.Module) -> None:
    """Reverse fuse_qkv_for_tp — splits fused weights back to separate projections.

    Call before every model save so checkpoints are compatible with anima_train.py.
    Call fuse_qkv_for_tp() again after saving to resume fast training.
    """
    import types
    from library.anima_models import Attention, LLMAdapterAttention
    original_compute_qkv = Attention.compute_qkv

    # Un-fuse main Attention modules
    for block in model.blocks:
        sa, ca = block.self_attn, block.cross_attn

        # Un-fuse self-attn: qkv_proj → q/k/v
        qkv_w = sa.qkv_proj.weight.data
        inner, in_f = qkv_w.shape[0] // 3, qkv_w.shape[1]
        dev, dt = qkv_w.device, qkv_w.dtype
        sa.q_proj = torch.nn.Linear(in_f, inner, bias=False, device=dev, dtype=dt)
        sa.k_proj = torch.nn.Linear(in_f, inner, bias=False, device=dev, dtype=dt)
        sa.v_proj = torch.nn.Linear(in_f, inner, bias=False, device=dev, dtype=dt)
        sa.q_proj.weight = torch.nn.Parameter(qkv_w[:inner].clone())
        sa.k_proj.weight = torch.nn.Parameter(qkv_w[inner:2*inner].clone())
        sa.v_proj.weight = torch.nn.Parameter(qkv_w[2*inner:].clone())
        del sa.qkv_proj
        sa.compute_qkv = types.MethodType(original_compute_qkv, sa)

        # Un-fuse cross-attn: kv_proj → k/v
        kv_w = ca.kv_proj.weight.data
        inner_kv, in_f_kv = kv_w.shape[0] // 2, kv_w.shape[1]
        ca.k_proj = torch.nn.Linear(in_f_kv, inner_kv, bias=False, device=dev, dtype=dt)
        ca.v_proj = torch.nn.Linear(in_f_kv, inner_kv, bias=False, device=dev, dtype=dt)
        ca.k_proj.weight = torch.nn.Parameter(kv_w[:inner_kv].clone())
        ca.v_proj.weight = torch.nn.Parameter(kv_w[inner_kv:].clone())
        del ca.kv_proj
        ca.compute_qkv = types.MethodType(original_compute_qkv, ca)

    # Un-fuse LLM adapter attention if present
    original_adapter_forward = LLMAdapterAttention.forward
    for name, module in model.named_modules():
        if not isinstance(module, LLMAdapterAttention):
            continue
        is_self = name.endswith(".self_attn")
        dev = next(module.parameters()).device
        dt  = next(module.parameters()).dtype
        if is_self and hasattr(module, "qkv_proj"):
            qkv_w = module.qkv_proj.weight.data
            inner, in_f = qkv_w.shape[0] // 3, qkv_w.shape[1]
            module.q_proj = torch.nn.Linear(in_f, inner, bias=False, device=dev, dtype=dt)
            module.k_proj = torch.nn.Linear(in_f, inner, bias=False, device=dev, dtype=dt)
            module.v_proj = torch.nn.Linear(in_f, inner, bias=False, device=dev, dtype=dt)
            module.q_proj.weight = torch.nn.Parameter(qkv_w[:inner].clone())
            module.k_proj.weight = torch.nn.Parameter(qkv_w[inner:2*inner].clone())
            module.v_proj.weight = torch.nn.Parameter(qkv_w[2*inner:].clone())
            del module.qkv_proj
            module.forward = types.MethodType(original_adapter_forward, module)
        elif not is_self and hasattr(module, "kv_proj"):
            kv_w = module.kv_proj.weight.data
            inner_kv, in_f_kv = kv_w.shape[0] // 2, kv_w.shape[1]
            module.k_proj = torch.nn.Linear(in_f_kv, inner_kv, bias=False, device=dev, dtype=dt)
            module.v_proj = torch.nn.Linear(in_f_kv, inner_kv, bias=False, device=dev, dtype=dt)
            module.k_proj.weight = torch.nn.Parameter(kv_w[:inner_kv].clone())
            module.v_proj.weight = torch.nn.Parameter(kv_w[inner_kv:].clone())
            del module.kv_proj
            module.forward = types.MethodType(original_adapter_forward, module)


# ---------------------------------------------------------------------------
# TP weight gather/restore for checkpoint saving
# ---------------------------------------------------------------------------

def _gather_tp_weights_for_save(dit: torch.nn.Module, tp_groups) -> list:
    """All-gather TP-sharded weights to full tensors so checkpoints are valid.

    Returns restore_info: list of (parent_module, attr_name, module, shard_data)
    so callers can restore the original sharded ColumnParallel/RowParallel modules
    after saving (the modules are kept alive via these references).
    """
    import torch.distributed as dist
    from wd_parallel.layers import ColumnParallelLinear, RowParallelLinear

    restore_info = []
    for name, module in dit.named_modules():
        is_col = isinstance(module, ColumnParallelLinear)
        is_row = isinstance(module, RowParallelLinear)
        if not (is_col or is_row):
            continue

        shard = module.weight.data.clone()
        w = module.weight.data.contiguous()
        tp_size = tp_groups.tp_size
        gathered = [torch.zeros_like(w) for _ in range(tp_size)]
        dist.all_gather(gathered, w, group=tp_groups.tp)

        gather_dim = 0 if is_col else 1
        full = torch.cat(gathered, dim=gather_dim)

        # Trim padding back to original unpadded size
        if is_col:
            orig = getattr(module, 'original_out_features', None)
            if orig is not None:
                full = full[:int(orig)]
        else:
            orig = getattr(module, 'original_in_features', None)
            if orig is not None:
                full = full[:, :int(orig)]

        module.weight.data = full

        parent_path, attr = name.rsplit('.', 1) if '.' in name else ('', name)
        parent = dit.get_submodule(parent_path) if parent_path else dit
        restore_info.append((parent, attr, module, shard))

    return restore_info


def _restore_tp_weight_shards(restore_info: list) -> None:
    """Restore ColumnParallel/RowParallel modules with their original shard weights."""
    for parent, attr, module, shard in restore_info:
        module.weight.data = shard
        setattr(parent, attr, module)


# ---------------------------------------------------------------------------
# TP spec
# ---------------------------------------------------------------------------

def _make_anima_tp_spec(
    sequence_parallel: bool = False,
    use_llm_adapter: bool = False,
    *,
    allow_padding: bool = True,
    padding_multiple: int = 1,
    fuse_qkv: bool = True,
) -> "wdp.ParallelSpec":
    """TP spec for Anima full finetune after fuse_qkv_for_tp() has been applied."""
    import wd_parallel as wdp
    sp = sequence_parallel
    col = lambda sp_flag: wdp.ColumnParallelSpec(
        sequence_parallel=sp_flag, seq_dim=1,
        allow_padding=allow_padding, padding_multiple=padding_multiple,
    )
    row = lambda sp_flag: wdp.RowParallelSpec(
        sequence_parallel=sp_flag, seq_dim=1,
        allow_padding=allow_padding, padding_multiple=padding_multiple,
    )
    packed_col = lambda sp_flag, parts: wdp.PackedColumnParallelSpec(
        sequence_parallel=sp_flag, seq_dim=1,
        packed_parts=parts,
        allow_padding=allow_padding, padding_multiple=padding_multiple,
    )
    if fuse_qkv:
        entries = {
            "blocks.*.self_attn.qkv_proj":     packed_col(sp, 3),
            "blocks.*.self_attn.output_proj":   row(sp),
            "blocks.*.cross_attn.q_proj":       col(sp),
            "blocks.*.cross_attn.kv_proj":      packed_col(False, 2),
            "blocks.*.cross_attn.output_proj":  row(sp),
            "blocks.*.mlp.layer1":              col(sp),
            "blocks.*.mlp.layer2":              row(sp),
        }
    else:
        entries = {
            "blocks.*.self_attn.q_proj":        col(sp),
            "blocks.*.self_attn.k_proj":        col(sp),
            "blocks.*.self_attn.v_proj":        col(sp),
            "blocks.*.self_attn.output_proj":   row(sp),
            "blocks.*.cross_attn.q_proj":       col(sp),
            "blocks.*.cross_attn.k_proj":       col(False),
            "blocks.*.cross_attn.v_proj":       col(False),
            "blocks.*.cross_attn.output_proj":  row(sp),
            "blocks.*.mlp.layer1":              col(sp),
            "blocks.*.mlp.layer2":              row(sp),
        }
    if use_llm_adapter:
        # Adapter T5 target sequence is REPLICATED on all TP ranks — always sp=False.
        if fuse_qkv:
            entries.update({
                "llm_adapter.blocks.*.self_attn.qkv_proj": packed_col(False, 3),
                "llm_adapter.blocks.*.self_attn.o_proj":   row(False),
                "llm_adapter.blocks.*.cross_attn.q_proj":  col(False),
                "llm_adapter.blocks.*.cross_attn.kv_proj": packed_col(False, 2),
                "llm_adapter.blocks.*.cross_attn.o_proj":  row(False),
                "llm_adapter.blocks.*.mlp.0":              col(False),
                "llm_adapter.blocks.*.mlp.2":              row(False),
            })
        else:
            entries.update({
                "llm_adapter.blocks.*.self_attn.q_proj":  col(False),
                "llm_adapter.blocks.*.self_attn.k_proj":  col(False),
                "llm_adapter.blocks.*.self_attn.v_proj":  col(False),
                "llm_adapter.blocks.*.self_attn.o_proj":  row(False),
                "llm_adapter.blocks.*.cross_attn.q_proj": col(False),
                "llm_adapter.blocks.*.cross_attn.k_proj": col(False),
                "llm_adapter.blocks.*.cross_attn.v_proj": col(False),
                "llm_adapter.blocks.*.cross_attn.o_proj": row(False),
                "llm_adapter.blocks.*.mlp.0":             col(False),
                "llm_adapter.blocks.*.mlp.2":             row(False),
            })
    return wdp.ParallelSpec(entries)


# ---------------------------------------------------------------------------
# TP-aware grad-norm clipping
# ---------------------------------------------------------------------------

def _make_tp_clip_grad_norm(tp_group):
    """Build a clip_grad_norm_ replacement that computes the global norm
    across TP ranks instead of each rank clipping to its local shard's norm.

    Ported verbatim (modulo naming) from
    anima_train_network_tensor_sequence_parallel.py's _tp_clip_grad_norm_,
    which is the LoRA-TP equivalent of this fix. Without this, FFT TP runs
    with --max_grad_norm set clip each rank's sharded params by a norm that
    only reflects that rank's slice, causing the sharded weights to diverge
    across ranks over time.
    """

    def _tp_clip_grad_norm_(
        parameters: Union[torch.Tensor, list],
        max_norm: float,
        norm_type: float = 2.0,
    ):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        if len(parameters) == 0:
            return torch.tensor(0.0)

        device = parameters[0].grad.device
        norm_type = float(norm_type)

        if norm_type == float('inf'):
            # inf norm: global max across all params and all TP ranks.
            # Sharded params: each rank holds a slice -> need all-reduce max.
            # Replicated params: identical on all ranks -> all-reduce max is a no-op but harmless.
            local_max = torch.tensor(
                max(p.grad.detach().abs().max().item() for p in parameters),
                device=device,
            )
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=tp_group)
            total_norm = local_max
        else:
            # Lp norm: sum of p-th powers, then take p-th root.
            #
            # Two kinds of params must be treated differently to avoid
            # counting the same gradient contribution multiple times:
            #
            #   TP-sharded (_tp_sharded=True):  each rank holds a unique shard
            #     -> squared norms across ranks sum to the full-weight squared norm
            #     -> must all-reduce (SUM) so every rank uses the global total.
            #
            #   Replicated (_tp_sharded absent/False):  every rank holds an
            #     identical copy (sync_replicated_grads already ran).
            #     -> simply adding all ranks' contributions would count the
            #       same gradient tp_size times, inflating the norm by
            #       sqrt(tp_size) and over-clipping by that factor.
            #     -> include only locally; no cross-rank reduction needed.
            sharded_acc = torch.zeros(1, device=device)
            replicated_acc = torch.zeros(1, device=device)
            for p in parameters:
                contrib = p.grad.detach().norm(norm_type).pow(norm_type)
                if getattr(p, '_tp_sharded', False):
                    sharded_acc += contrib
                else:
                    replicated_acc += contrib

            # All-reduce only the sharded contribution
            dist.all_reduce(sharded_acc, op=dist.ReduceOp.SUM, group=tp_group)
            total_norm = (sharded_acc + replicated_acc).pow(1.0 / norm_type)

        # Clip using the global norm (identical on all ranks after all-reduce)
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for p in parameters:
            p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))

        return total_norm.item()

    return _tp_clip_grad_norm_


# ---------------------------------------------------------------------------
# TP-aware trainer mixin — shared by plain AnimaTrainer and MuonAnimaTrainer
# ---------------------------------------------------------------------------

class TPSPTrainerMixin:
    """Hook overrides that add TP+SP support on top of either AnimaTrainer
    (anima_train.py) or MuonAnimaTrainer (anima_train_muon.py).

    Both base classes expose an identical hook surface (on_train_begin,
    apply_model_parallelism, prepare_dit_with_accelerator, sync_gradients,
    before_save, after_save, on_train_end, pre_process_batch — see either
    file's docstring), so this mixin is combined with each base via
    AnimaTrainerTPSP / MuonAnimaTrainerTPSP below rather than duplicating the
    override bodies per optimizer.
    """

    def __init__(self):
        super().__init__()
        self.tp_groups       = None
        self.tp_config       = None
        self.tp_active       = False
        self.use_sp          = False
        self.train_dit       = True
        self._tp_save_restore = None

    # --- Hook: QKV fusion + TP sharding (after DiT load, before optimizer) ---

    def apply_model_parallelism(self, args, dit):
        if self.tp_groups is None or self.tp_groups.tp_size <= 1:
            return dit

        import wd_parallel as wdp

        self.train_dit  = args.learning_rate != 0
        use_llm_adapter = getattr(dit, 'use_llm_adapter', False)
        fuse_qkv        = not getattr(args, 'no_fuse_qkv', False)

        # 1. Infer padding geometry (handles non-divisible tp_degree)
        tp_geometry = _infer_anima_tp_padding_geometry(dit, self.tp_groups.tp_size)

        # 2. Fuse QKV (idempotent, includes LLM adapter)
        fused_count = fuse_qkv_for_tp(dit, include_llm_adapter=use_llm_adapter) if fuse_qkv else 0

        # 3. Build TP spec and apply sharding
        tp_spec = _make_anima_tp_spec(
            self.use_sp,
            use_llm_adapter=use_llm_adapter,
            allow_padding=True,
            padding_multiple=tp_geometry["head_dim"],
            fuse_qkv=fuse_qkv,
        )
        dit = wdp.apply_parallelism(dit, tp_spec, self.tp_config, self.tp_groups)
        self.tp_active = True

        # 4. Fix n_heads on each Attention to match local shard width
        n_attn_fixed = _fixup_attention_heads_for_tp(dit)

        # 5. Mark replicated cross-attn KV layers to skip input grad (text encoder is frozen)
        n_no_input_grad = _mark_replicated_context_layers_no_input_grad(dit, text_encoder_frozen=True)

        # 6. SP group: set on model so forward scatter/gather uses the right process group
        if self.use_sp:
            dit._tp_sp_group = self.tp_groups.tp

        # 7. Reuse the forward all-gather for the wgrad computation, saving one
        # collective per trainable column-parallel layer per backward.
        # "auto" enables it iff gradient checkpointing is on (memory-free
        # there; without it, holding the gathered activation costs extra VRAM).
        reuse_mode = getattr(args, 'tp_reuse_wgrad_gather', 'auto')
        if reuse_mode == 'auto':
            reuse_wgrad_gather = bool(getattr(args, 'gradient_checkpointing', False))
        else:
            reuse_wgrad_gather = reuse_mode == 'on'
        n_reuse = wdp.set_wgrad_gather_reuse(dit, reuse_wgrad_gather)

        logger.info(
            f"TP sharding applied: tp_degree={self.tp_groups.tp_size}, sp={self.use_sp}, "
            f"llm_adapter={use_llm_adapter}, fuse_qkv={fuse_qkv}, "
            f"fused_attention_modules={fused_count}, "
            f"attention_modules_patched={n_attn_fixed}, "
            f"replicated_context_no_input_grad={n_no_input_grad}, "
            f"model_width={tp_geometry['model_channels']}->{tp_geometry['padded_width']}, "
            f"local_width={tp_geometry['local_width']}, local_heads={tp_geometry['local_heads']}, "
            f"head_dim={tp_geometry['head_dim']}, padding_added={tp_geometry['padding_added']}, "
            f"reuse_wgrad_gather={reuse_wgrad_gather} (mode={reuse_mode}, {n_reuse} layers)"
        )

        # 8. Optional full-model forward check
        if getattr(args, 'tp_verify_model_forward', False):
            from tp_sp_verify import run_all_checks as _tp_verify
            _tp_verify(dit=dit, network=None, groups=self.tp_groups, use_sp=self.use_sp)

        # 9. NaN diagnostics
        n_nan_hooks = _register_tp_nan_hooks(dit)
        logger.info(f"TP NaN diagnostic hooks registered on {n_nan_hooks} layer(s)")

        return dit

    # --- Hook: skip Accelerator DDP wrapping for TP ---

    def prepare_dit_with_accelerator(self, accelerator, dit, is_swapping_blocks):
        if not self.tp_active:
            return super().prepare_dit_with_accelerator(accelerator, dit, is_swapping_blocks)

        # Monkeypatch clip_grad_norm_ to compute the global norm across TP
        # ranks. Without this, each rank clips TP-sharded params using only
        # its local shard's norm, causing sharded weights to drift apart.
        accelerator.clip_grad_norm_ = _make_tp_clip_grad_norm(self.tp_groups.tp)

        # TP handles its own communication — DDP wrapping would conflict.
        if is_swapping_blocks:
            dit.move_to_device_except_swap_blocks(accelerator.device)
            dit.prepare_block_swap_before_forward()
        else:
            dit.to(accelerator.device)
        return dit

    # --- Hook: broadcast batch from rank 0 so all TP ranks see the same data ---

    @staticmethod
    def _broadcast_tensor(t, tp_group, device):
        """Broadcast a tensor from rank 0 to all TP ranks, handling shape mismatches."""
        if t is None:
            return None
        t = t.to(device)
        shape_t = torch.tensor(list(t.shape), dtype=torch.int64, device=device)
        dist.broadcast(shape_t, src=0, group=tp_group)
        canonical = torch.Size(shape_t.tolist())
        if t.shape != canonical:
            t = torch.zeros(canonical, dtype=t.dtype, device=device)
        t = t.contiguous()
        dist.broadcast(t, src=0, group=tp_group)
        return t

    def pre_process_batch(self, batch: dict, accelerator) -> dict:
        """Broadcast batch from rank 0 so all TP ranks see the same data.

        TP requires all ranks to process the SAME batch. Without this, each rank
        independently draws from the DataLoader (same seed but diverged Python RNG),
        causing ColumnParallelLinear allgathers to receive unequal-sized inputs.
        """
        if not self.tp_active or not dist.is_initialized():
            return batch

        tp_group = self.tp_groups.tp
        dev      = accelerator.device

        # Latents (shape differs across resolution buckets)
        if "latents" in batch and batch["latents"] is not None:
            batch["latents"] = self._broadcast_tensor(batch["latents"], tp_group, dev)

        # Per-sample loss weights
        if "loss_weights" in batch and batch["loss_weights"] is not None:
            batch["loss_weights"] = self._broadcast_tensor(batch["loss_weights"], tp_group, dev)

        # Cached TE outputs
        te_list = batch.get("text_encoder_outputs_list", None)
        if te_list is not None:
            batch["text_encoder_outputs_list"] = [
                self._broadcast_tensor(te, tp_group, dev) for te in te_list
            ]
        else:
            # On-the-fly encoding: broadcast input IDs so all ranks encode same prompts
            ids_list = batch.get("input_ids_list", None)
            if ids_list is not None:
                batch["input_ids_list"] = [
                    self._broadcast_tensor(ids, tp_group, dev) for ids in ids_list
                ]

        # Alpha masks (optional, masked loss)
        if "alpha_masks" in batch and batch["alpha_masks"] is not None:
            batch["alpha_masks"] = self._broadcast_tensor(batch["alpha_masks"], tp_group, dev)

        return batch

    # --- Hook: sync non-sharded param gradients across TP ranks ---

    def sync_gradients(self, dit):
        if self.tp_active and self.tp_groups.tp_size > 1:
            wdp.sync_replicated_grads(dit, self.tp_groups.tp)

    # --- Hook: gather TP shards + unfuse QKV before save ---

    def before_save(self, dit):
        if self.tp_active and self.train_dit:
            # Gather all sharded weights to full tensors so the checkpoint has
            # correct shapes. Must be a collective (all ranks participate in
            # all_gather), so call on every rank before the rank-0-only write.
            self._tp_save_restore = _gather_tp_weights_for_save(dit, self.tp_groups)
            unfuse_qkv_from_tp(dit)

    # --- Hook: re-fuse QKV and restore TP shards so training continues ---

    def after_save(self, dit, train_dit):
        if self.tp_active and train_dit:
            use_llm_adapter = getattr(dit, 'use_llm_adapter', False)
            fuse_qkv_for_tp(dit, include_llm_adapter=use_llm_adapter)
            # Restore original ColumnParallel/RowParallel modules with sharded
            # weights. fuse_qkv_for_tp created plain nn.Linear for qkv_proj/
            # kv_proj; setattr replaces them back with the TP-wired originals.
            if self._tp_save_restore:
                _restore_tp_weight_shards(self._tp_save_restore)
                self._tp_save_restore = None
            dit.requires_grad_(train_dit)

    # --- Hook: gather + final unfuse before end-of-training saves ---

    def on_train_end(self, dit):
        if self.tp_active and self.train_dit:
            _gather_tp_weights_for_save(dit, self.tp_groups)
            unfuse_qkv_from_tp(dit)


# ---------------------------------------------------------------------------
# Concrete trainers — mixin combined with each optimizer's base trainer
# ---------------------------------------------------------------------------

class AnimaTrainerTPSP(TPSPTrainerMixin, AnimaTrainer):
    """TP+SP with the standard optimizer path (AdamW/Prodigy/etc, no Muon)."""
    pass


class MuonAnimaTrainerTPSP(TPSPTrainerMixin, MuonAnimaTrainer):
    """TP+SP with Muon+AdamW hybrid optimizer support.

    The Muon side is handled entirely by muon_optimizer.TPMuonWithAuxAdam,
    which MuonAnimaTrainer's optimizer-construction dispatch selects
    automatically once args._tp_groups is set (see __main__ below) — no
    override needed here beyond inheriting MuonAnimaTrainer's optimizer logic
    alongside this mixin's TP/SP hooks.
    """
    pass


# ---------------------------------------------------------------------------
# Parser — base + Muon args + TP/SP additions
# ---------------------------------------------------------------------------

def setup_parser() -> argparse.ArgumentParser:
    # anima_train_muon's parser is a strict superset of anima_train's (base
    # args + --use_muon/--muon_*); building on it means this script accepts
    # identical flags whether or not --use_muon is passed, so existing
    # non-Muon TP configs work unchanged.
    parser = _muon_setup_parser()
    parser.add_argument(
        "--tp_degree", type=int, default=1,
        help="Tensor Parallel degree. 1=disabled (plain single-GPU). Requires torchrun --nproc_per_node=N.",
    )
    parser.add_argument(
        "--tp_backend", type=str, default="auto",
        choices=["auto", "gloo", "cuda_direct", "nccl"],
        help="Distributed backend for TP. Use cuda_direct on Windows, nccl on WSL/Linux.",
    )
    parser.add_argument(
        "--sequence_parallel", action="store_true", default=False,
        help="Enable Sequence Parallel alongside TP (requires --tp_degree >= 2).",
    )
    parser.add_argument(
        "--tp_verify_model_forward", action="store_true",
        help="Run the expensive full-DiT TP/SP forward diagnostic before training.",
    )
    parser.add_argument(
        "--no_fuse_qkv", action="store_true",
        help="Disable internal fused QKV/KV projections (for debugging).",
    )
    parser.add_argument(
        "--tp_reuse_wgrad_gather", type=str, default="auto",
        choices=["auto", "on", "off"],
        help=(
            "Reuse the forward all-gather for the weight-gradient computation "
            "in TP column-parallel layers instead of re-gathering in backward "
            "(saves one collective per trainable layer per backward, ~11%% "
            "faster in benchmarks). 'auto' (default) enables it iff "
            "--gradient_checkpointing is set (memory-free there); 'on'/'off' "
            "force the setting regardless of checkpointing."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Required for async TP collective overlap: CUDA scheduler must honour
    # kernel-submission order so async AG/RS runs before the dependent matmul.
    _cuda_max_conn = os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS")
    if _cuda_max_conn is None:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    elif _cuda_max_conn != "1":
        logger.warning(
            f"CUDA_DEVICE_MAX_CONNECTIONS={_cuda_max_conn!r} (expected '1'). "
            "Async TP collective overlap may not work correctly."
        )

    parser = setup_parser()
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    tp_degree = int(getattr(args, "tp_degree", 1))
    use_muon  = muon_optimizer.is_muon_enabled(args)

    # tp_degree=1: run as plain AnimaTrainer/MuonAnimaTrainer, no TP init
    if tp_degree <= 1:
        (MuonAnimaTrainer() if use_muon else AnimaTrainer()).train(args)
        sys.exit(0)

    if not _WDP_AVAILABLE:
        raise RuntimeError("wd_parallel is required for TP but could not be imported.")

    # --fused_backward_pass / --blockwise_fused_optimizers step and zero grads
    # inside per-parameter hooks that run BEFORE sync_replicated_grads(), which
    # would silently desync TP-replicated params (LayerNorm/AdaLN/embedders).
    # Configs are reused across TP/DDP/FSDP/Muon runs, so warn + fall back
    # instead of crashing on a stale flag from a different mode.
    if getattr(args, 'fused_backward_pass', False) or getattr(args, 'blockwise_fused_optimizers', False):
        logger.warning(
            "--fused_backward_pass/--blockwise_fused_optimizers are incompatible with TP+SP "
            "(they step/zero grads before sync_replicated_grads runs, silently desyncing "
            "replicated params). Disabling both for this run."
        )
        args.fused_backward_pass = False
        args.blockwise_fused_optimizers = False

    tp_backend = wdp.activate_backend(getattr(args, "tp_backend", "auto"))
    dist.init_process_group(backend=tp_backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    if world_size != tp_degree:
        raise ValueError(
            f"tp_degree={tp_degree} must match torchrun world_size={world_size}"
        )

    use_sp     = getattr(args, 'sequence_parallel', False)
    tp_config  = wdp.ParallelConfig(tp=True, sp=use_sp, tp_degree=tp_degree)
    tp_groups  = wdp.init_dist(tp_config)
    logger.info(
        f"TP+SP initialized: rank={tp_groups.tp_rank}/{tp_groups.tp_size}, "
        f"backend={tp_backend}, sp={use_sp}"
    )

    # Wrap train_util save/resume helpers for per-rank state folders
    save_utils.install_parallel_state_wrappers(
        train_util_module=train_util,
        huggingface_util_module=huggingface_util,
        parallel_rank=tp_groups.tp_rank,
        parallel_size=tp_groups.tp_size,
        process_group=tp_groups.tp,
        backend=tp_backend,
        logger=logger,
        mode="tp_sp",
    )

    # Pre-training sanity check (collective math + TP layer equivalence)
    from tp_sp_verify import run_all_checks as _tp_verify
    _tp_verify(dit=None, network=None, groups=tp_groups, use_sp=use_sp)

    # Spoof Accelerator state so all TP ranks enter save/load hooks (which
    # contain TP collectives) and DDP wrapping + DistributedSampler are skipped.
    # Both AcceleratorState AND PartialState must be updated: AcceleratorState
    # copies from PartialState on every construction, so patching only one leaks.
    _orig_prepare_accelerator = train_util.prepare_accelerator
    def _tp_prepare_accelerator(args):
        from accelerate.state import AcceleratorState, PartialState
        from accelerate.utils import DistributedType
        acc = _orig_prepare_accelerator(args)
        spoof = {
            "distributed_type":    DistributedType.NO,
            "process_index":       0,
            "local_process_index": 0,
            "num_processes":       1,
        }
        PartialState._shared_state.update(spoof)
        AcceleratorState._shared_state.update(spoof)
        acc.prepare_data_loader = lambda dl, **_: dl
        logger.info(
            f"TP accelerator spoof: rank={tp_groups.tp_rank}/{world_size}, "
            f"distributed_type={acc.state.distributed_type}, "
            f"num_processes={acc.num_processes}, "
            f"is_main={acc.is_main_process}"
        )
        return acc
    train_util.prepare_accelerator = _tp_prepare_accelerator

    # Muon optimizer construction (muon_optimizer.build_anima_muon_optimizer)
    # reads args._tp_groups to select TPMuonWithAuxAdam (exact gather-NS on
    # TP-sharded params) instead of the DDP-oriented MuonWithAuxAdam, whose
    # round-robin all_gather assumes replicated params and would corrupt TP
    # shards. Force-disable Muon's own distributed all-gather path too —
    # TPMuonWithAuxAdam handles TP communication itself.
    if use_muon:
        args._tp_groups = tp_groups
        args.muon_disable_distributed_allgather = True

    # Create trainer and inject TP state (before trainer.train() so hooks see it)
    trainer = (MuonAnimaTrainerTPSP() if use_muon else AnimaTrainerTPSP())
    trainer.tp_config = tp_config
    trainer.tp_groups = tp_groups
    trainer.use_sp    = use_sp

    try:
        trainer.train(args)
    finally:
        wdp.destroy_dist()
