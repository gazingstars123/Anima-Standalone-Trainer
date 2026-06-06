"""
Muon optimizer integration for Anima full finetuning.

This module wraps the upstream ``muon-optimizer`` package (Muon / SingleDeviceMuon /
MuonWithAuxAdam / SingleDeviceMuonWithAuxAdam) and re-buckets the per-component
parameter groups produced by ``anima_train_utils.get_anima_param_groups`` into a
single ``use_muon=True/False`` group list that the upstream optimizer can consume.

It does NOT touch the upstream Muon source, the upstream ``anima_train.py``, or any
file in ``library/train_util.py``. The construction functions here are called from
``anima_train_muon.py`` (a copy of ``anima_train.py`` with a diverged optimizer
section). When ``--use_muon`` is not set, the construction falls back to the
standard ``train_util.get_optimizer`` path so behavior is identical to the
upstream script.

References
----------
- Keller Jordan, "Muon: An optimizer for hidden layers in neural networks"
  https://kellerjordan.github.io/posts/muon/
- Liu et al., "Muon is Scalable for LLM Training" (Kimi Moonlight), arXiv:2502.16982
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import OrderedDict, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

import library.train_util as train_util
from library import anima_train_utils

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


def add_muon_arguments(parser: argparse.ArgumentParser) -> None:
    """Append Muon-specific CLI flags to an existing parser.

    The flags are prefixed ``--muon_*`` so they cannot collide with existing
    Anima flags. None of them are required; the master switch ``--use_muon``
    defaults to False, in which case the rest of the Muon options are ignored
    and the script behaves identically to ``anima_train.py``.
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
            "Multiplier applied to --muon_lr. For Anima finetune (base AdamW lr ~2e-6), "
            "0.05 yields an effective Muon lr of ~1e-3 which is a reasonable starting point. "
            "Tune per workload."
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
            "Decoupled weight decay for Muon 2D params. Kimi Moonlight (arxiv 2502.16982) "
            "shows WD is critical for Muon at scale. 0.01 is a safe default; 0.0 disables it."
        ),
    )
    muon_group.add_argument(
        "--muon_ns_steps",
        type=int,
        default=5,
        help="Newton-Schulz orthogonalization iterations. 5 is the standard.",
    )
    muon_group.add_argument(
        "--muon_adam_lr",
        type=float,
        default=None,
        help=(
            "Learning rate for the AdamW half. If omitted, the per-component LRs from "
            "anima_train_utils.get_anima_param_groups (e.g. --self_attn_lr, --mlp_lr, "
            "--mod_lr, --llm_adapter_lr) are used, with the overall --learning_rate "
            "scaling the base group."
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
            "Required under FSDP/FSDP2 (where grads are already reduce-scattered). "
            "Auto-detected under accelerate's FSDP2 path; set manually otherwise."
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


# ---------------------------------------------------------------------------
# Distributed / FSDP2 detection
# ---------------------------------------------------------------------------


def _detect_training_mode(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    """Inspect the runtime to decide which Muon class to instantiate.

    Returns a dict with keys:
        - use_distributed_muon: bool  (use MuonWithAuxAdam with built-in all_gather)
        - world_size: int
        - is_fsdp2: bool
        - reason: str                   (human-readable explanation for logging)
    """
    dist_available = torch.distributed.is_available() and torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size() if dist_available else 1
    is_fsdp2 = False
    if args is not None:
        # accelerate exposes the underlying FSDP plugin via state; the most reliable
        # public signal is the attribute added in anima_train.py at line 705.
        is_fsdp2 = bool(getattr(args, "_fsdp2_active", False))

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

    return dict(use_distributed_muon=use_dist, world_size=world_size, is_fsdp2=is_fsdp2, reason=reason)


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
    name = _param_full_name(p)
    return name.startswith("llm_adapter.") or "llm_adapter." in name


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
    if policy == "all_2d":
        return True
    if policy == "custom":
        # 'custom' = all 2D except llm_adapter (which is already excluded above)
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


def split_anima_params_for_muon(
    dit: torch.nn.Module,
    raw_param_groups: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Re-bucket DiT parameters for the Muon+AdamW hybrid.

    Walks ``dit.named_parameters()`` directly so we know the canonical component
    for each param (base / self_attn / cross_attn / mlp / mod / llm_adapter).
    This avoids relying on ``raw_param_groups`` carrying a ``name`` key (which
    ``anima_train_utils.get_anima_param_groups`` does NOT include in its output).

    Per-component LRs are taken from ``raw_param_groups`` (which is the output
    of ``get_anima_param_groups``) by matching the per-component LRs the user
    passed via ``--self_attn_lr``, ``--mlp_lr``, etc. If a component is missing
    from ``raw_param_groups`` (e.g. frozen, lr=0), its params are skipped.
    """
    # Build a canonical per-component LR map from the output of
    # get_anima_param_groups. The order of that output is:
    #   (base, self_attn, cross_attn, mlp, mod, llm_adapter)
    # but lr=0 and empty groups are filtered out, so we detect them by walking
    # named_parameters and using the same per-component LR if it appears in any
    # group with matching params.
    # Simplest: just re-derive component LRs by calling get_anima_param_groups
    # with the same kwargs. The helper handles lr=0 freezing internally.
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


def _get_muon_classes():
    """Late import so the rest of the codebase doesn't require muon-optimizer."""
    from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
    return MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam


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

    The class does NOT pass the param groups to ``super().__init__``'s
    registration loop (which would double-register). Instead it initializes
    the base-class bookkeeping manually (``defaults``, hook dicts, ``state``,
    ``param_groups``) and then *replaces* ``self.param_groups`` with a
    property that delegates to ``self._inner.param_groups``.
    """

    def __init__(self, inner: torch.optim.Optimizer):
        # We have to bypass the base-class add_param_group loop, so we replicate
        # the minimal bookkeeping that Optimizer.__init__ sets up.
        # All these are read by PyTorch internals.
        object.__setattr__(self, "defaults", dict(inner.defaults) if inner.defaults else {})
        object.__setattr__(self, "_optimizer_step_pre_hooks", OrderedDict())
        object.__setattr__(self, "_optimizer_step_post_hooks", OrderedDict())
        object.__setattr__(self, "_optimizer_state_dict_pre_hooks", OrderedDict())
        object.__setattr__(self, "_optimizer_state_dict_post_hooks", OrderedDict())
        object.__setattr__(self, "_optimizer_load_state_dict_pre_hooks", OrderedDict())
        object.__setattr__(self, "_optimizer_load_state_dict_post_hooks", OrderedDict())
        # We don't have a Tensor param list to register; emulate the empty fallback.
        object.__setattr__(self, "state", defaultdict(dict))
        object.__setattr__(self, "_warned_capturable_if_run_uncaptured", True)
        # The hook function below is what Optimizer.__init__ calls; we have to
        # patch our own step into the pre-hook list so torch's
        # _use_grad_context_aware will be honored.
        self._patch_step_function()
        self._inner = inner

    # -- delegation properties --
    @property
    def param_groups(self):
        return self._inner.param_groups

    # -- protocol methods --
    def step(self, closure=None):
        # Set zero grads for any param that lacks a grad. This guards against
        # the missing None-check in both the Muon and AdamW branches of the
        # upstream v0.1.0's step().
        patched = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                    patched.append(p)
        try:
            import inspect
            step_fn = self._inner.step
            try:
                sig = inspect.signature(step_fn)
            except (TypeError, ValueError):
                sig = None
            if sig is not None and len(sig.parameters) >= 2:
                return step_fn(closure)
            return step_fn()
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
    *,
    base_lr: Optional[float] = None,
    self_attn_lr: Optional[float] = None,
    cross_attn_lr: Optional[float] = None,
    mlp_lr: Optional[float] = None,
    mod_lr: Optional[float] = None,
    llm_adapter_lr: Optional[float] = None,
) -> torch.optim.Optimizer:
    """Build the Muon+AdamW hybrid optimizer for an Anima DiT.

    IMPORTANT: this function does NOT call ``anima_train_utils.get_anima_param_groups``
    to compute component LRs, because that helper has a side effect of calling
    ``requires_grad_(False)`` on any param whose component has lr=0. If we let it
    run, params the user wants trainable (e.g. ``--llm_adapter_lr 2e-6``) would
    be frozen before our split sees them. Instead, we read per-component LRs
    from ``args`` (or the explicit kwargs) and pass them straight to
    ``split_anima_params_for_muon`` which is the only place we want to make
    the freeze decision.
    """
    mixed_groups = split_anima_params_for_muon(dit, [], args)

    mode = _detect_training_mode(args)
    MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam = _get_muon_classes()

    if mode["use_distributed_muon"]:
        optimizer_class = MuonWithAuxAdam
    else:
        optimizer_class = SingleDeviceMuonWithAuxAdam
    logger.info(
        f"[muon] using {optimizer_class.__name__} ({mode['reason']})"
    )

    _validate_group_schema(mixed_groups)

    inner = optimizer_class(mixed_groups)
    optimizer = _SafeMuonWrapper(inner)
    return optimizer


def _validate_group_schema(groups: List[Dict[str, Any]]) -> None:
    """Strip non-schema keys before passing groups to the upstream optimizer class.

    The upstream ``MuonWithAuxAdam``/``SingleDeviceMuonWithAuxAdam`` assert an exact
    key set per group in their __init__, so we must remove any extra keys we
    accumulated (such as ``name`` and ``ns_steps``) before construction. The
    stripping is silent because the keys are present purely for our own logging.
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

    Falls back to ``train_util.get_optimizer`` unchanged otherwise. The signature
    mirrors the upstream factory so the call site is a one-line swap.
    """
    if not is_muon_enabled(args) or not train_dit or dit is None:
        return train_util.get_optimizer(args, trainable_params)

    if getattr(args, "fused_backward_pass", False):
        raise ValueError(
            "--use_muon is incompatible with --fused_backward_pass (which currently "
            "requires Adafactor). Disable --fused_backward_pass when using Muon."
        )
    if getattr(args, "blockwise_fused_optimizers", False):
        raise ValueError(
            "--use_muon is incompatible with --blockwise_fused_optimizers. "
            "Muon needs full visibility over all 2D params for Newton-Schulz."
        )
    if getattr(args, "deepspeed", False):
        raise ValueError(
            "--use_muon is incompatible with --deepspeed (DeepSpeed ZeRO has its own "
            "optimizer-subclassing protocol that the upstream muon package does not support)."
        )

    optimizer = build_anima_muon_optimizer(
        dit,
        args,
        base_lr=getattr(args, "learning_rate", None),
        self_attn_lr=getattr(args, "self_attn_lr", None),
        cross_attn_lr=getattr(args, "cross_attn_lr", None),
        mlp_lr=getattr(args, "mlp_lr", None),
        mod_lr=getattr(args, "mod_lr", None),
        llm_adapter_lr=getattr(args, "llm_adapter_lr", None),
    )
    return "MuonWithAuxAdam", str(_detect_training_mode(args)["reason"]), optimizer
