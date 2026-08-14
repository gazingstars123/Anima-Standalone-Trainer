from __future__ import annotations

import argparse
import inspect
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch

import library.train_util as train_util

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


def add_muon_arguments(parser: argparse.ArgumentParser) -> None:
    """Append Muon-specific CLI flags to an existing parser.
    The flags are prefixed ``--muon_*``
    """

    parser.add_argument(
        "--use_muon",
        action="store_true",
        help=(
            "Master switch: replace the standard optimizer with a "
            "Muon+AdamW hybrid (Keller Jordan, https://kellerjordan.github.io/posts/muon/). "
            "Muon optimizes 2D DiT weight matrices; AdamW handles 1D params, "
            "embeddings, adaln_modulation, and the LLM adapter."
        ),
    )

    muon_group = parser.add_argument_group("Muon optimizer")
    muon_group.add_argument(
        "--muon_lr",
        type=float,
        default=0.02,
        help="Base Muon learning rate in spectral-norm units (Keller Jordan default 0.02).",
    )
    muon_group.add_argument(
        "--muon_lr_scale",
        type=float,
        default=0.05,
        help=(
            "Multiplier applied to --muon_lr to compute the effective Muon learning rate."
        ),
    )
    muon_group.add_argument(
        "--muon_momentum",
        type=float,
        default=0.95,
        help="SGD-momentum beta used inside the Muon update. Nesterov is always on.",
    )
    muon_group.add_argument(
        "--muon_weight_decay",
        type=float,
        default=0.01,
        help=(
            "Decoupled weight decay for Muon 2D params. Default 0.01."
        ),
    )
    muon_group.add_argument(
        "--muon_ns_steps",
        type=int,
        default=5,
        help="Newton-Schulz orthogonalization iterations. Default 5",
    )
    muon_group.add_argument(
        "--muon_adam_lr",
        type=float,
        default=None,
        help=(
            "Learning rate for the AdamW half. Default from AdamW lr"
        ),
    )
    muon_group.add_argument(
        "--muon_adam_betas",
        type=str,
        default="0.9,0.95",
        help="Betas for the AdamW half (comma-separated).",
    )
    muon_group.add_argument(
        "--muon_adam_eps",
        type=float,
        default=1e-8,
        help="Eps for the AdamW half.",
    )
    muon_group.add_argument(
        "--muon_param_filter",
        type=str,
        choices=("self_attn_mlp_cross", "all_2d", "custom"),
        default="self_attn_mlp_cross",
        help=(
            "Which 2D parameters receive Muon. "
            "'self_attn_mlp_cross' (default): 2D weights inside self_attn, cross_attn, "
            "mlp, and DiT 'base' (block stems/embed/final) groups. 1D params, adaln_modulation, "
            "and llm_adapter always go to AdamW regardless of this flag. "
            "'all_2d': every trainable 2D parameter, including those in llm_adapter. "
            "'custom': same as 'all_2d' minus llm_adapter."
        ),
    )
    muon_group.add_argument(
        "--muon_disable_distributed_allgather",
        action="store_true",
        help=(
            "Disable the all_gather Muon does inside step() when world_size > 1. "
            "Required under FSDP/FSDP2"
        ),
    )
    muon_group.add_argument(
        "--muon_disable_for_llm_adapter",
        action="store_true",
        default=True,
        help="Always force llm_adapter weights to AdamW regardless of --muon_param_filter. Default: True.",
    )
    muon_group.add_argument(
        "--muon_disable_for_adaln",
        action="store_true",
        default=True,
        help="Always force adaln_modulation weights to AdamW regardless of --muon_param_filter. Default: True.",
    )
    muon_group.add_argument(
        "--muon_fp32_sensitive",
        action="store_true",
        help="Keep precision-sensitive AdamW params (adaln_modulation, q_norm/k_norm) in fp32.",
    )


# ---------------------------------------------------------------------------
# Distributed / FSDP2 detection
# ---------------------------------------------------------------------------


def _detect_training_mode(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    """Inspect the runtime to decide which Muon class to instantiate.

    Returns a dict with keys:
        - use_distributed_muon: bool  (use MuonWithAuxAdam with built-in all_gather)
        - use_tp_muon: bool           (use TPMuonWithAuxAdam — Tensor Parallel active)
        - world_size: int
        - is_fsdp2: bool
        - reason: str                   (human-readable explanation for logging)
    """
    dist_available = torch.distributed.is_available() and torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size() if dist_available else 1
    is_fsdp2 = False
    tp_groups = getattr(args, "_tp_groups", None) if args is not None else None
    if args is not None:
        is_fsdp2 = bool(getattr(args, "_fsdp2_active", False))

    if tp_groups is not None and getattr(tp_groups, "tp_size", 1) > 1:
        return dict(
            use_distributed_muon=False,
            use_tp_muon=True,
            world_size=world_size,
            is_fsdp2=is_fsdp2,
            reason=(
                f"Tensor Parallel active (tp_size={tp_groups.tp_size}); using "
                "TPMuonWithAuxAdam (exact gather-NS on TP-sharded params)"
            ),
        )

    explicit_disable = bool(args and getattr(args, "muon_disable_distributed_allgather", False))
    use_dist = (world_size > 1) and (not is_fsdp2) and (not explicit_disable)

    if world_size == 1:
        reason = "single-process (world_size=1)"
    elif is_fsdp2:
        reason = "FSDP2 detected; using SingleDeviceMuonWithAuxAdam (FSDP2 handles its own reduce-scatter)"
    elif explicit_disable:
        reason = "--muon_disable_distributed_allgather set"
    else:
        reason = f"DDP active (world_size={world_size}); using MuonWithAuxAdam with built-in all_gather"

    return dict(
        use_distributed_muon=use_dist, use_tp_muon=False,
        world_size=world_size, is_fsdp2=is_fsdp2, reason=reason,
    )


# ---------------------------------------------------------------------------
# Parameter splitting
# ---------------------------------------------------------------------------


def _param_full_name(p: torch.nn.Parameter) -> str:
    """Best-effort fully-qualified parameter name (set by get_anima_param_groups)."""
    return getattr(p, "original_name", "") or ""


def _is_adaln_param(p: torch.nn.Parameter) -> bool:
    name = _param_full_name(p)
    return "adaln_modulation" in name


def _is_llm_adapter_param(p: torch.nn.Parameter) -> bool:
    return "llm_adapter." in _param_full_name(p)


def _should_use_muon(p: torch.nn.Parameter, group_name: str, args: argparse.Namespace) -> bool:
    """Decision: should this specific parameter be optimized with Muon?

    Rules, in order of precedence:
      1. Any 1D tensor (bias, gain, norm) -> never.
      2. 4D conv-shaped tensor (ndim==4) -> never, because the installed
         muon-optimizer==0.1.0 has a known bug in 4D handling: ``muon_update``
         flattens to 2D but the inner ``step()`` does not reshape the 2D
         update back to 4D before ``p.add_()``. Routing 4D weights to Muon
         would crash. The DiT in Anima has no 4D conv weights, so this is a
         no-op for the current model.
      3. adaln_modulation params -> never (always AdamW).
      4. llm_adapter params -> never, unless --muon_disable_for_llm_adapter is False.
      5. Otherwise, apply the --muon_param_filter policy.
    """
    if p.ndim < 2:
        return False
    if p.ndim == 4:
        # See rule 2 above. Logged once per call by split_anima_params_for_muon.
        return False
    if getattr(args, "muon_disable_for_adaln", True) and _is_adaln_param(p):
        return False
    if getattr(args, "muon_disable_for_llm_adapter", True) and _is_llm_adapter_param(p):
        return False

    policy = getattr(args, "muon_param_filter", "self_attn_mlp_cross")
    if policy in ("all_2d", "custom"):
        # 'custom' = all 2D except llm_adapter, which is already excluded above.
        return True
    # default: 'self_attn_mlp_cross'
    return group_name in ("base", "self_attn", "cross_attn", "mlp")


def _classify_param(name: str) -> str:
    """Return the canonical Anima component name for a fully-qualified param name.

    The canonical names match the order in anima_train_utils.get_anima_param_groups:
    'base' | 'self_attn' | 'cross_attn' | 'mlp' | 'mod' | 'llm_adapter'.
    """
    if "llm_adapter" in name:
        return "llm_adapter"
    if "adaln_modulation" in name:
        return "mod"
    if ".self_attn." in name:
        return "self_attn"
    if ".cross_attn." in name:
        return "cross_attn"
    if ".mlp." in name:
        return "mlp"
    return "base"


def upcast_sensitive_params_to_fp32(dit: torch.nn.Module) -> int:
    # muon: upcast precision-sensitive components to fp32
    n = 0
    for name, p in dit.named_parameters():
        if p.requires_grad and ("adaln_modulation" in name or ".q_norm." in name or ".k_norm." in name):
            p.data = p.data.float()
            n += 1
    return n


def split_anima_params_for_muon(
    dit: torch.nn.Module,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Re-bucket DiT parameters for the Muon+AdamW hybrid.
    """
    component_lrs: Dict[str, float] = {}
    component_lrs["base"] = float(getattr(args, "learning_rate", 0.0))
    for key in ("self_attn_lr", "cross_attn_lr", "mlp_lr", "mod_lr", "llm_adapter_lr"):
        v = getattr(args, key, None)
        if v is None:
            v = float(getattr(args, "learning_rate", 0.0))
        component_lrs[key.removesuffix("_lr")] = float(v)

    # Now walk params.
    muon_buckets: Dict[str, List[torch.nn.Parameter]] = {}
    adam_buckets: Dict[str, Dict[str, Any]] = {}
    skipped_frozen = 0

    for name, p in dit.named_parameters():
        if not isinstance(p, torch.nn.Parameter):
            continue
        if not p.requires_grad:
            skipped_frozen += 1
            continue

        # Tag for the per-param-name path used by the helper.
        p.original_name = name

        comp = _classify_param(name)

        # Skip components with lr=0 (frozen)
        if component_lrs.get(comp, 0.0) == 0.0:
            p.requires_grad_(False)
            skipped_frozen += 1
            continue

        if _should_use_muon(p, comp, args):
            bucket_key = f"muon_{comp}"
            muon_buckets.setdefault(bucket_key, []).append(p)
        else:
            bucket_key = f"adam_{comp}"
            if bucket_key not in adam_buckets:
                adam_buckets[bucket_key] = {
                    "params": [],
                    "lr": component_lrs[comp],
                    "weight_decay": 0.0,
                }
            adam_buckets[bucket_key]["params"].append(p)

    betas = _parse_betas(getattr(args, "muon_adam_betas", "0.9,0.95"))
    adam_eps = float(getattr(args, "muon_adam_eps", 1e-8))
    adam_lr_override = getattr(args, "muon_adam_lr", None)

    out: List[Dict[str, Any]] = []

    # ---- Muon groups ----
    muon_lr_base = float(args.muon_lr) * float(getattr(args, "muon_lr_scale", 1.0))
    base_lr = float(getattr(args, "learning_rate", 1.0)) or 1.0
    for bucket_key in sorted(muon_buckets.keys()):
        comp = bucket_key.removeprefix("muon_")
        comp_lr = component_lrs.get(comp, base_lr)
        lr_ratio = comp_lr / base_lr
        effective_lr = muon_lr_base * lr_ratio
        out.append(
            {
                "params": muon_buckets[bucket_key],
                "lr": effective_lr,
                "momentum": float(args.muon_momentum),
                "weight_decay": float(args.muon_weight_decay),
                "use_muon": True,
                "name": bucket_key,
            }
        )

    # ---- Adam groups ----
    for bucket_key in sorted(adam_buckets.keys()):
        bucket = adam_buckets[bucket_key]
        lr = float(adam_lr_override) if adam_lr_override is not None else float(bucket["lr"])
        out.append(
            {
                "params": bucket["params"],
                "lr": lr,
                "betas": betas,
                "eps": adam_eps,
                "weight_decay": float(bucket["weight_decay"]),
                "use_muon": False,
                "name": bucket_key,
            }
        )

    total_muon = sum(len(b) for b in muon_buckets.values())
    total_adam = sum(len(b["params"]) for b in adam_buckets.values())
    logger.info(
        f"[muon] param split: {total_muon} 2D params -> Muon, "
        f"{total_adam} 1D/embed/mod/llm -> AdamW (skipped {skipped_frozen} frozen)"
    )
    if muon_buckets:
        for k, v in muon_buckets.items():
            logger.info(f"  muon/{k}: {len(v)} params")
    if adam_buckets:
        for k, v in adam_buckets.items():
            logger.info(f"  adam/{k}: {len(v['params'])} params")

    return out


def _parse_betas(s: str):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--muon_adam_betas expects 'beta1,beta2', got: {s}")
    return (float(parts[0]), float(parts[1]))


# ---------------------------------------------------------------------------
# Optimizer construction
# ---------------------------------------------------------------------------


_MUON_INSTALL_HINT = (
    "Muon requires the muon-optimizer package: pip install muon-optimizer\n"
    "Note the name: `pip install muon` installs an unrelated omics library, not this optimizer."
)


def _get_muon_classes():
    """Late import so the rest of the codebase doesn't require muon-optimizer."""
    try:
        from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
    except ImportError as e:
        raise ImportError(_MUON_INSTALL_HINT) from e
    return MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam


_sharded_aux_adam_cls = None


def _get_sharded_aux_adam_class():
    """Build (once) the DDP Muon variant whose aux-AdamW branch is sharded.
    """
    global _sharded_aux_adam_cls
    if _sharded_aux_adam_cls is not None:
        return _sharded_aux_adam_cls

    import torch.distributed as dist
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
    try:
        from muon import MuonWithAuxAdam, muon_update, adam_update
    except ImportError as e:
        raise ImportError(_MUON_INSTALL_HINT) from e

    class ShardedAuxAdamMuonWithAuxAdam(MuonWithAuxAdam):
        @torch.no_grad()
        def step(self):
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            for group in self.param_groups:
                if group["use_muon"]:
                    params = group["params"]
                    params_pad = params + [torch.empty_like(params[-1])] * (len(params) % world_size)
                    pending = None
                    for base_i in range(len(params))[::world_size]:
                        if base_i + rank < len(params):
                            p = params[base_i + rank]
                            state = self.state[p]
                            if len(state) == 0:
                                state["momentum_buffer"] = torch.zeros_like(p)
                            update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                            p.add_(update, alpha=-group["lr"])
                        if pending is not None:
                            pending.wait()
                        pending = dist.all_gather(
                            params_pad[base_i:base_i + world_size], params_pad[base_i + rank], async_op=True
                        )
                    if pending is not None:
                        pending.wait()
                else:
                    params = group["params"]
                    # Step only the params this rank owns (round-robin by index).
                    for i in range(rank, len(params), world_size):
                        p = params[i]
                        state = self.state[p]
                        if len(state) == 0:
                            state["exp_avg"] = torch.zeros_like(p)
                            state["exp_avg_sq"] = torch.zeros_like(p)
                            state["step"] = 0
                        state["step"] += 1
                        update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                             state["step"], group["betas"], group["eps"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update, alpha=-group["lr"])

                    for owner in range(world_size):
                        owned = [params[i] for i in range(owner, len(params), world_size)]
                        if not owned:
                            continue
                        by_dtype = {}
                        for p in owned:
                            by_dtype.setdefault(str(p.dtype), []).append(p)
                        for dtype_key in sorted(by_dtype.keys()):
                            plist = by_dtype[dtype_key]
                            flat = _flatten_dense_tensors([p.data for p in plist])
                            dist.broadcast(flat, src=owner)
                            if rank != owner:
                                for p, synced in zip(plist, _unflatten_dense_tensors(flat, [p.data for p in plist])):
                                    p.data.copy_(synced)

    _sharded_aux_adam_cls = ShardedAuxAdamMuonWithAuxAdam
    return _sharded_aux_adam_cls


# ---------------------------------------------------------------------------
# TP-aware Muon: exact gather-NS on TP-sharded weight matrices
# ---------------------------------------------------------------------------

def _muon_local_pre_ns_update(grad: torch.Tensor, momentum_buffer: torch.Tensor, beta: float, nesterov: bool = True) -> torch.Tensor:
    momentum_buffer.lerp_(grad, 1 - beta)
    return grad.lerp_(momentum_buffer, beta) if nesterov else momentum_buffer


def _muon_orthogonalize(update_2d: torch.Tensor, ns_steps: int) -> torch.Tensor:
    """NS + scale tail of upstream muon.muon_update, given a full 2D matrix."""
    from muon import zeropower_via_newtonschulz5
    out = zeropower_via_newtonschulz5(update_2d, steps=ns_steps)
    out = out * max(1, update_2d.size(-2) / update_2d.size(-1)) ** 0.5
    return out


def _tp_muon_orthogonalized_shard(
    update_shard: torch.Tensor,
    p: torch.nn.Parameter,
    tp_group,
    ns_steps: int,
) -> torch.Tensor:
    from wd_parallel.layers import (
        merge_column_shards, merge_row_shards,
        _shard_colwise, _shard_packed_colwise, _shard_rowwise,
    )

    world_size = p._tp_world_size
    rank = p._tp_rank
    kind = p._tp_shard_kind
    allow_padding = p._tp_allow_padding
    padding_multiple = p._tp_padding_multiple

    gathered = [torch.empty_like(update_shard) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, update_shard.contiguous(), group=tp_group)

    if kind == "row":
        full = merge_row_shards(gathered, original_in_features=p._tp_original_in_features, dim=1)
        full_ns = _muon_orthogonalize(full, ns_steps)
        return _shard_rowwise(full_ns, rank, world_size, allow_padding=allow_padding, padding_multiple=padding_multiple)

    if kind == "col":
        full = merge_column_shards(gathered, original_out_features=p._tp_original_out_features, dim=0)
        full_ns = _muon_orthogonalize(full, ns_steps)
        return _shard_colwise(full_ns, rank, world_size, allow_padding=allow_padding, padding_multiple=padding_multiple)

    if kind == "packed_col":
        packed_parts = p._tp_packed_parts
        original_part_size = p._tp_original_part_size
        full = merge_column_shards(
            gathered, original_out_features=p._tp_original_out_features, dim=0,
            packed_parts=packed_parts,
            original_part_size=original_part_size,
            padded_part_size=p._tp_padded_part_size,
        )
        parts = full.split(original_part_size, dim=0)
        ns_parts = [_muon_orthogonalize(part, ns_steps) for part in parts]
        full_ns = torch.cat(ns_parts, dim=0)
        return _shard_packed_colwise(
            full_ns, rank, world_size, packed_parts,
            allow_padding=allow_padding, padding_multiple=padding_multiple,
        )

    raise ValueError(f"unknown TP shard kind {kind!r} on param (expected col/packed_col/row)")


class TPMuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups: List[Dict[str, Any]], tp_group, ns_steps: int = 5):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())
        self._tp_group = tp_group
        self._ns_steps = int(ns_steps)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        from muon import muon_update, adam_update

        for group in self.param_groups:
            if group["use_muon"]:
                beta = group["momentum"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    if getattr(p, "_tp_sharded", False):
                        pre_ns = _muon_local_pre_ns_update(p.grad, state["momentum_buffer"], beta)
                        update = _tp_muon_orthogonalized_shard(pre_ns, p, self._tp_group, self._ns_steps)
                    else:
                        update = muon_update(p.grad, state["momentum_buffer"], beta=beta, ns_steps=self._ns_steps)
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad, state["exp_avg"], state["exp_avg_sq"],
                        state["step"], group["betas"], group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class _SafeMuonWrapper(torch.optim.Optimizer):
    """Defensive wrapper around the upstream Muon+AdamW class.

    The installed ``muon-optimizer==0.1.0`` is missing a ``p.grad is None`` guard
    in both the Muon and AdamW branches of the single-device
    ``MuonWithAuxAdam.step()``: if a param didn't receive a backward pass
    (e.g. an unused branch in a conditional forward), ``p.grad is None`` and
    the inner update functions raise ``TypeError`` deep inside
    ``momentum.lerp_``.

    This wrapper subclasses ``torch.optim.Optimizer`` so it type-checks as a
    proper optimizer (PyTorch's ``lr_scheduler`` and ``accelerate.prepare``
    both require it). The actual param groups, state, and step logic live on
    the wrapped inner optimizer; we delegate to it and only add the missing
    grad=None guard.
    """

    def __init__(self, inner: torch.optim.Optimizer):
        object.__setattr__(self, "defaults", dict(inner.defaults) if inner.defaults else {})
        object.__setattr__(self, "_optimizer_step_pre_hooks", OrderedDict())
        object.__setattr__(self, "_optimizer_step_post_hooks", OrderedDict())
        object.__setattr__(self, "_optimizer_state_dict_pre_hooks", OrderedDict())
        object.__setattr__(self, "_optimizer_state_dict_post_hooks", OrderedDict())
        object.__setattr__(self, "_optimizer_load_state_dict_pre_hooks", OrderedDict())
        object.__setattr__(self, "_optimizer_load_state_dict_post_hooks", OrderedDict())
        object.__setattr__(self, "_warned_capturable_if_run_uncaptured", True)

        self._patch_step_function()
        self._inner = inner

        try:
            sig = inspect.signature(inner.step)
            self._inner_step_takes_closure = len(sig.parameters) >= 2
        except (TypeError, ValueError):
            self._inner_step_takes_closure = False

    # -- delegation properties --
    @property
    def param_groups(self):
        return self._inner.param_groups

    @property
    def state(self):
        return self._inner.state

    # -- protocol methods --
    def step(self, closure=None):
        # Set zero grads for any param that lacks a grad.
        patched = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                    patched.append(p)
        try:
            if self._inner_step_takes_closure:
                return self._inner.step(closure)
            return self._inner.step()
        finally:
            for p in patched:
                p.grad = None

    def zero_grad(self, set_to_none=True):
        return self._inner.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self._inner.state_dict()

    def load_state_dict(self, state_dict):
        return self._inner.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        return self._inner.add_param_group(param_group)

    @property
    def inner(self):
        return self._inner


def build_anima_muon_optimizer(
    dit: torch.nn.Module,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    """Build the Muon+AdamW hybrid optimizer for an Anima DiT.
    """
    mixed_groups = split_anima_params_for_muon(dit, args)

    mode = _detect_training_mode(args)
    _strip_non_schema_keys(mixed_groups)

    if mode["use_tp_muon"]:
        tp_group = args._tp_groups.tp
        ns_steps = int(getattr(args, "muon_ns_steps", 5))
        logger.info(f"[muon] using TPMuonWithAuxAdam ({mode['reason']})")
        inner = TPMuonWithAuxAdam(mixed_groups, tp_group, ns_steps=ns_steps)
        return _SafeMuonWrapper(inner)

    MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam = _get_muon_classes()
    # Distributed DDP path always uses the sharded-aux variant: same Muon branch,
    # but AdamW state (norms/adaln/llm_adapter) is round-robin sharded across
    # ranks instead of fully replicated, mirroring how Muon momentum already is.
    optimizer_class = _get_sharded_aux_adam_class() if mode["use_distributed_muon"] else SingleDeviceMuonWithAuxAdam
    logger.info(f"[muon] using {optimizer_class.__name__} ({mode['reason']})")

    inner = optimizer_class(mixed_groups)
    return _SafeMuonWrapper(inner)


def _strip_non_schema_keys(groups: List[Dict[str, Any]]) -> None:
    """Strip non-schema keys before passing groups to the upstream optimizer class.
    """
    muon_required = {"params", "lr", "momentum", "weight_decay", "use_muon"}
    adam_required = {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}

    for g in groups:
        required = muon_required if g.get("use_muon") else adam_required
        for k in list(g.keys()):
            if k not in required:
                del g[k]


# ---------------------------------------------------------------------------
# Public API for anima_train_muon.py
# ---------------------------------------------------------------------------


def is_muon_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "use_muon", False))


def get_optimizer(args, trainable_params, *, train_dit: bool = True, dit: Optional[torch.nn.Module] = None):
    """Drop-in replacement for ``train_util.get_optimizer`` that routes to Muon
    when ``--use_muon`` is set and ``train_dit`` is True.
    """
    if not is_muon_enabled(args) or not train_dit or dit is None:
        return train_util.get_optimizer(args, trainable_params)

    if getattr(args, "fused_backward_pass", False):
        raise ValueError(
            "--use_muon is incompatible with --fused_backward_pass"
        )
    if getattr(args, "blockwise_fused_optimizers", False):
        raise ValueError(
            "--use_muon is incompatible with --blockwise_fused_optimizers."
        )
    if getattr(args, "deepspeed", False):
        raise ValueError(
            "--use_muon is incompatible with --deepspeed"
        )

    optimizer = build_anima_muon_optimizer(dit, args)
    return type(optimizer.inner).__name__, _detect_training_mode(args)["reason"], optimizer


# ---------------------------------------------------------------------------
# DDP State Save Fix
# ---------------------------------------------------------------------------


def gather_muon_state_before_save(optimizer) -> None:
    """Fix incomplete Muon optimizer state under DDP multi-GPU before saving.
    """
    inner = optimizer
    seen_ids = set()
    while id(inner) not in seen_ids:
        seen_ids.add(id(inner))
        wrapped = getattr(inner, "_inner", None) or getattr(inner, "inner", None) or getattr(inner, "optimizer", None)
        if wrapped is None:
            break
        inner = wrapped
    if inner is optimizer:
        return

    if "Muon" not in type(inner).__name__:
        return

    if "SingleDevice" in type(inner).__name__ or type(inner).__name__ == "TPMuonWithAuxAdam":
        return

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return

    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return

    rank = torch.distributed.get_rank()
    aux_sharded = type(inner).__name__ == "ShardedAuxAdamMuonWithAuxAdam"

    # Global param order, identical on all ranks
    ordered = []
    for group in inner.param_groups:
        use_muon = bool(group.get("use_muon"))
        for pos, p in enumerate(group["params"]):
            ordered.append((use_muon, pos, p))

    # Each rank contributes only the shards it owns. Tensors are moved to CPU
    local_entries = {}
    for gidx, (use_muon, pos, p) in enumerate(ordered):
        if not (use_muon or aux_sharded) or pos % world_size != rank:
            continue
        param_state = inner.state.get(p)
        if param_state:
            local_entries[gidx] = {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in param_state.items()
            }

    # gather_object (not all_gather_object): only rank 0 needs the combined state,
    gathered = [None] * world_size if rank == 0 else None
    torch.distributed.gather_object(local_entries, gathered, dst=0)

    if rank == 0:
        for src_rank in range(1, world_size):
            for gidx, param_state in gathered[src_rank].items():
                inner.state[ordered[gidx][2]] = param_state

    # Make sure rank 0 finished merging before save_state proceeds.
    torch.distributed.barrier()


def scatter_muon_state_after_load(optimizer) -> None:
    """Undo gather_muon_state_before_save()'s effect after accelerator.load_state().
    """
    inner = optimizer
    seen_ids = set()
    while id(inner) not in seen_ids:
        seen_ids.add(id(inner))
        wrapped = getattr(inner, "_inner", None) or getattr(inner, "inner", None) or getattr(inner, "optimizer", None)
        if wrapped is None:
            break
        inner = wrapped
    if inner is optimizer:
        return

    if "Muon" not in type(inner).__name__:
        return
    if "SingleDevice" in type(inner).__name__ or type(inner).__name__ == "TPMuonWithAuxAdam":
        return

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return

    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return

    rank = torch.distributed.get_rank()
    aux_sharded = type(inner).__name__ == "ShardedAuxAdamMuonWithAuxAdam"

    for group in inner.param_groups:
        use_muon = bool(group.get("use_muon"))
        if not use_muon and not aux_sharded:
            continue
        for pos, p in enumerate(group["params"]):
            if pos % world_size != rank and p in inner.state:
                del inner.state[p]
