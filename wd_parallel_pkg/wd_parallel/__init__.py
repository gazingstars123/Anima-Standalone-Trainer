"""
wd_parallel - Windows-native TP+SP training framework.

Designed for multi-GPU training on Windows (no NCCL required).
Can use cuda_direct SHM zero-copy transport (~22-28 GB/s), gloo, or NCCL
when the caller selects that backend before initialization.
All collectives call dist.* directly - no DTensor functional collectives.

Quick start:
    import wd_parallel as wdp

    config  = wdp.ParallelConfig(tp=True, sp=True)
    dist.init_process_group()                    # caller/training script owns backend
    groups  = wdp.init_dist(config)              # build TP/DP sub-groups

    # Or force a backend explicitly:
    # backend = wdp.activate_backend("cuda_direct")  # cuda_direct, gloo, nccl, auto
    # dist.init_process_group(backend=backend)

    model = MyModel(...)
    spec  = wdp.ParallelSpec({
        "blocks.*.attn.qkv":    wdp.ColumnParallelSpec(sequence_parallel=True),
        "blocks.*.attn.out":    wdp.RowParallelSpec(),
        "blocks.*.mlp.fc1":     wdp.ColumnParallelSpec(sequence_parallel=True),
        "blocks.*.mlp.fc2":     wdp.RowParallelSpec(),
    })
    model = wdp.apply_parallelism(model, spec, config, groups)

    for batch in loader:
        loss = model(batch)
        loss.backward()
        wdp.sync_replicated_grads(model, groups.tp)
        optimizer.step()

    wdp.destroy_dist()
"""

from .config import ParallelConfig
from .backends import activate_backend
from .initialize import ProcessGroups, init_dist, destroy_dist
from .collectives import (
    gather_from_sp_region,
    reduce_scatter_to_sp_region,
    copy_to_tp_region,
    copy_to_tp_region_no_input_grad,
    comm_timer,
    GlobalMemoryBuffer,
    reset_nan_diagnostics,
    set_nan_diagnostics_enabled,
    gather_from_sp_region_and_trim,
    pad_to_world_size,
    split_along_dim_with_padding,
)
from .layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    merge_column_shards,
    merge_row_shards,
    trim_padded_features,
    trim_packed_padded_features,
)
from .apply import (
    ParallelSpec,
    ColumnParallelSpec,
    PackedColumnParallelSpec,
    RowParallelSpec,
    apply_parallelism,
    sync_replicated_grads,
    set_wgrad_gather_reuse,
)
from .fsdp import wrap_fsdp

__version__ = "0.1.0"

__all__ = [
    # Config
    "ParallelConfig",
    # Backend
    "activate_backend",
    # Init
    "ProcessGroups", "init_dist", "destroy_dist",
    # Collectives
    "gather_from_sp_region",
    "reduce_scatter_to_sp_region", "copy_to_tp_region",
    "copy_to_tp_region_no_input_grad",
    "comm_timer", "GlobalMemoryBuffer", "reset_nan_diagnostics",
    "set_nan_diagnostics_enabled",
    "gather_from_sp_region_and_trim", "pad_to_world_size",
    "split_along_dim_with_padding",
    # Layers
    "ColumnParallelLinear", "RowParallelLinear",
    "merge_column_shards", "merge_row_shards",
    "trim_padded_features", "trim_packed_padded_features",
    # Apply
    "ParallelSpec", "ColumnParallelSpec", "PackedColumnParallelSpec", "RowParallelSpec",
    "apply_parallelism", "sync_replicated_grads", "set_wgrad_gather_reuse",
    # FSDP
    "wrap_fsdp",
]
