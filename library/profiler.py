from tqdm import tqdm
import os
import time
import torch
import torch.distributed as dist

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False

class StepProfiler:
    def __init__(self, accelerator, enabled=False, profile_microbatch=False):
        self.accelerator = accelerator
        self.enabled = enabled
        self.profile_microbatch = profile_microbatch and enabled

        self._cum_fwd = 0.0
        self._cum_bwd = 0.0
        self._cum_comm = 0.0
        self._t0_step = None
        self._t0 = None  # start of current micro-batch
        self._t1 = None  # end of fwd
        self._t2 = None  # end of bwd
        self._t3 = None  # end of comm

        self._peak_vram = 0  # peak real VRAM 
        self._nvml_handle = None
        if enabled and _NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                local_index = accelerator.device.index or 0
                cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
                if cvd:
                    ids = [int(x) for x in cvd.split(",") if x.strip() != ""]
                    nvml_index = ids[local_index] if local_index < len(ids) else local_index
                else:
                    nvml_index = local_index
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
            except Exception:
                self._nvml_handle = None

        # Microbatch tracking
        self._mb_index = 0
        self._mb_lines = []  # collected lines to print at step end

        # Running averages (averaged across ranks)
        self._avg_count = 0
        self._avg_wall = 0.0
        self._avg_fwd  = 0.0
        self._avg_bwd  = 0.0
        self._avg_comm = 0.0
        self._avg_opt  = 0.0
        self._avg_vram = 0.0
        # Collective running averages (cuda_direct only)
        self._avg_ag_ms  = 0.0  # allgather
        self._avg_rs_ms  = 0.0  # reduce_scatter
        self._avg_ar_ms  = 0.0  # allreduce
        self._avg_a2a_ms = 0.0  # alltoall

        # Memory snapshot
        self._snapshot_step = None

        try:
            if dist.is_initialized() and dist.get_world_size() > 1:
                self._world_size = dist.get_world_size()
                self._is_rank0   = dist.get_rank() == 0
            else:
                self._world_size = accelerator.num_processes
                self._is_rank0   = accelerator.is_main_process
        except Exception:
            self._world_size = accelerator.num_processes
            self._is_rank0   = accelerator.is_main_process

        # CommTimer: auto-detected from the default process group if using
        # cuda_direct backend on multi-GPU. None on single GPU or other backends.
        self._comm_timer = None
        if enabled and self._world_size > 1:
            try:
                import torch.distributed.distributed_c10d as _c10d
                _pg = _c10d._get_default_group()
                if hasattr(_pg, "comm_timer"):
                    self._comm_timer = _pg.comm_timer
            except Exception:
                pass

    def _nvml_poll(self):
        if self._nvml_handle is None:
            return
        try:
            used = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle).used
            self._peak_vram = max(self._peak_vram, used)
        except Exception:
            pass

    def request_snapshot(self, step: int):
        """Schedule a full memory snapshot for the given global step.

        The snapshot captures every allocation/free with stack traces.
        Output: memory_snapshot_step{N}.pickle — load at pytorch.org/memory_viz
        """
        self._snapshot_step = step

    # ------------------------------------------------------------------
    # Phase hooks
    # ------------------------------------------------------------------

    def on_batch_start(self):
        if not self.enabled: return
        torch.cuda.synchronize()
        now = time.perf_counter()
        if self._t0_step is None:
            self._t0_step = now
            self._mb_index = 0
            self._mb_lines = []
            if self._snapshot_step is not None:
                torch.cuda.memory._record_memory_history(max_entries=100_000)
        self._t0 = now
        self._mb_index += 1
        self._nvml_poll()
        if self._comm_timer is not None:
            self._comm_timer.set_context("fwd")

    def on_fwd_done(self):
        if not self.enabled: return
        torch.cuda.synchronize()
        self._t1 = time.perf_counter()
        self._cum_fwd += (self._t1 - self._t0) * 1000
        self._nvml_poll()
        if self._comm_timer is not None:
            self._comm_timer.set_context("bwd")

    def on_bwd_done(self):
        if not self.enabled: return
        torch.cuda.synchronize()
        self._t2 = time.perf_counter()
        self._cum_bwd += (self._t2 - self._t1) * 1000
        # Default t3 to t2 so that if on_comm_done isn't called, comm time is 0
        self._t3 = self._t2
        self._nvml_poll()
        if self._comm_timer is not None:
            self._comm_timer.set_context("comm")

        if self.profile_microbatch:
            ms_fwd = (self._t1 - self._t0) * 1000
            ms_bwd = (self._t2 - self._t1) * 1000
            self._mb_lines.append(
                f"    microbatch {self._mb_index}: fwd={ms_fwd:.1f}ms  bwd={ms_bwd:.1f}ms"
            )

    def on_comm_done(self):
        if not self.enabled: return
        torch.cuda.synchronize()
        now = time.perf_counter()
        self._cum_comm += (now - self._t2) * 1000  # accumulate, since this fires every micro-batch, not just the last
        self._t3 = now
        self._nvml_poll()

    def on_step_done(self, global_step):
        if not self.enabled: return

        # Only print summary when accumulation is complete
        if not self.accelerator.sync_gradients: return

        torch.cuda.synchronize()
        t4 = time.perf_counter()
        self._nvml_poll()

        # Dump snapshot if this was the requested step
        if self._snapshot_step is not None:
            fname = f"memory_snapshot_step{global_step + 1}.pickle"
            try:
                torch.cuda.memory._dump_snapshot(fname)
                tqdm.write(f"[PROFILE] memory snapshot saved → {fname}  (load at pytorch.org/memory_viz)")
            except Exception as e:
                tqdm.write(f"[PROFILE] memory snapshot failed: {e}")
            torch.cuda.memory._record_memory_history(enabled=None)
            self._snapshot_step = None

        ms_comm = self._cum_comm  # accumulated across all micro-batches, not just the last
        ms_opt  = (t4 - self._t3) * 1000  # optimizer.step() only runs once, so the last micro-batch's t3 is correct here
        ms_wall = (t4 - self._t0_step) * 1000
        ms_fwd  = self._cum_fwd
        ms_bwd  = self._cum_bwd

        comm_events = []
        ag_ms = rs_ms = ar_ms = a2a_ms = 0.0
        if self._comm_timer is not None:
            comm_events = self._comm_timer.drain()
            self._comm_timer.set_context("")
            for e in comm_events:
                if   e.op == "allgather":       ag_ms  += e.duration_ms
                elif e.op == "reduce_scatter":  rs_ms  += e.duration_ms
                elif e.op == "allreduce":        ar_ms  += e.duration_ms
                elif e.op == "alltoall":         a2a_ms += e.duration_ms

        def _mb(b): return b / 1024 ** 2

        metrics = torch.tensor([
            ms_wall, ms_fwd, ms_bwd, ms_comm, ms_opt, _mb(self._peak_vram),
            ag_ms, rs_ms, ar_ms, a2a_ms,
        ], device=self.accelerator.device)

        try:
            if dist.is_initialized() and dist.get_world_size() > 1:
                gather_list = [torch.zeros_like(metrics) for _ in range(self._world_size)]
                dist.all_gather(gather_list, metrics)
                gathered = torch.cat(gather_list)
            else:
                gathered = self.accelerator.gather(metrics)
        except Exception:
            gathered = self.accelerator.gather(metrics)

        # Discard the allgather fired above so it doesn't bleed into the next
        if self._comm_timer is not None:
            self._comm_timer.drain()

        if self._is_rank0:
            N = 10
            num_ranks = self._world_size
            rank_data = [gathered[i*N : (i+1)*N].tolist() for i in range(num_ranks)]

            # Running average: mean across ranks then accumulate.
            mean = [sum(r[k] for r in rank_data) / num_ranks for k in range(N)]
            self._avg_count += 1
            self._avg_wall  += mean[0]
            self._avg_fwd   += mean[1]
            self._avg_bwd   += mean[2]
            self._avg_comm  += max(r[3] for r in rank_data)
            self._avg_opt   += mean[4]
            self._avg_vram  += mean[5]
            self._avg_ag_ms  += mean[6]
            self._avg_rs_ms  += mean[7]
            self._avg_ar_ms  += mean[8]
            self._avg_a2a_ms += mean[9]
            n = self._avg_count

            output_lines = [f"[PROFILE step {global_step + 1}]"]

            if num_ranks == 1:
                # Single GPU — no rank label, no collective section
                m = rank_data[0]
                output_lines.append(
                    f"  wall={m[0]:.1f}ms  "
                    f"fwd={m[1]:.1f}ms  "
                    f"bwd={m[2]:.1f}ms  "
                    f"opt={m[4]:.1f}ms  "
                    f"peak_vram={m[5]:.0f}MB"
                )
                output_lines.append(
                    f"  avg/{n}:  wall={self._avg_wall/n:.1f}ms  "
                    f"fwd={self._avg_fwd/n:.1f}ms  "
                    f"bwd={self._avg_bwd/n:.1f}ms  "
                    f"opt={self._avg_opt/n:.1f}ms  "
                    f"peak_vram={self._avg_vram/n:.0f}MB"
                )
            else:
                # Multi-GPU — per-rank lines + avg + optional collective breakdown
                for i, m in enumerate(rank_data):
                    output_lines.append(
                        f"  rank {i}: wall={m[0]:.1f}ms  "
                        f"fwd={m[1]:.1f}ms  "
                        f"bwd={m[2]:.1f}ms  "
                        f"comm={m[3]:.1f}ms  "
                        f"opt={m[4]:.1f}ms  "
                        f"peak_vram={m[5]:.0f}MB"
                    )
                output_lines.append(
                    f"  avg/{n}:  wall={self._avg_wall/n:.1f}ms  "
                    f"fwd={self._avg_fwd/n:.1f}ms  "
                    f"bwd={self._avg_bwd/n:.1f}ms  "
                    f"comm(max)={self._avg_comm/n:.1f}ms  "
                    f"opt={self._avg_opt/n:.1f}ms  "
                    f"peak_vram={self._avg_vram/n:.0f}MB"
                )

                # Collective breakdown — only when cuda_direct is active.
                # Uses rank 0's local events
                has_coll = any(mean[k] > 0 for k in range(6, 10))
                if has_coll and comm_events:
                    try:
                        from cuda_direct_backend.comm_timer import CommTimer
                        ctx_stats = CommTimer.summary(
                            comm_events, group_by=("context", "op"))
                        # Group into three display buckets
                        buckets = {"fwd": [], "bwd": [], "comm": [], "": []}
                        for (ctx, op), s in ctx_stats.items():
                            bucket = buckets.get(ctx, buckets[""])
                            bucket.append(
                                f"{op}={s['total_ms']:.0f}ms"
                                f"×{s['count']}"
                            )
                        coll_parts = []
                        for label, key in [("fwd", "fwd"), ("bwd", "bwd"), ("comm", "comm")]:
                            if buckets[key]:
                                coll_parts.append(f"{label}[{', '.join(buckets[key])}]")
                        if coll_parts:
                            output_lines.append(
                                f"  collectives (rank 0): " + "  ".join(coll_parts))
                        # Collective running averages
                        avg_parts = []
                        if self._avg_ag_ms  > 0: avg_parts.append(f"allgather={self._avg_ag_ms/n:.0f}ms")
                        if self._avg_rs_ms  > 0: avg_parts.append(f"reduce_scatter={self._avg_rs_ms/n:.0f}ms")
                        if self._avg_ar_ms  > 0: avg_parts.append(f"allreduce={self._avg_ar_ms/n:.0f}ms")
                        if self._avg_a2a_ms > 0: avg_parts.append(f"alltoall={self._avg_a2a_ms/n:.0f}ms")
                        if avg_parts:
                            output_lines.append(
                                f"  coll avg/{n}: " + "  ".join(avg_parts))
                    except ImportError:
                        pass

            if self.profile_microbatch and self._mb_lines:
                output_lines.append("  microbatches (rank 0):")
                output_lines.extend(self._mb_lines)

            tqdm.write("\n" + "\n".join(output_lines))


        self._cum_fwd = 0.0
        self._cum_bwd = 0.0
        self._cum_comm = 0.0
        self._peak_vram = 0
        self._mb_index = 0
        self._mb_lines = []
        self._t0_step = None
