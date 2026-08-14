from concurrent.futures import ThreadPoolExecutor
import gc
import time
from typing import Any, Optional, Union, Callable, Tuple
import torch
import torch.nn as nn

try:
    import torch.compiler
    disable_compiler = torch.compiler.disable
except (ImportError, AttributeError):
    try:
        import torch._dynamo
        disable_compiler = torch._dynamo.disable
    except (ImportError, AttributeError):
        def disable_compiler(fn):
            return fn


# Keep these functions here for portability, and private to avoid confusion with the ones in device_utils.py
def _clean_memory_on_device(device: torch.device):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()

    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


def _synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# Offloading frees/refills each weight's GPU storage in place
_param_cpu_buffers: dict[int, torch.Tensor] = {}
_param_cpu_buffers_pinned: dict[int, bool] = {}


def _weight_params(layer: nn.Module):
    for module in layer.modules():
        weight = getattr(module, "weight", None)
        if weight is not None:
            yield weight


def _cpu_buffer_for(weight: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    key = id(weight)
    buf = _param_cpu_buffers.get(key)
    if buf is None:
        buf = torch.empty_like(weight, device="cpu")
        pinned = True
        try:
            buf = buf.pin_memory()
        except RuntimeError:
            pinned = False
        _param_cpu_buffers[key] = buf
        _param_cpu_buffers_pinned[key] = pinned
    return buf, _param_cpu_buffers_pinned[key]


def _offload_weights(layer: nn.Module, non_blocking: bool) -> list:
    # copy weights to CPU buffers; caller frees storages after the copies complete
    offloaded = []
    for weight in _weight_params(layer):
        if weight.device.type != "cuda" or weight.data.untyped_storage().nbytes() == 0:
            continue
        buf, pinned = _cpu_buffer_for(weight)
        copy_non_blocking = non_blocking and pinned
        if copy_non_blocking and weight.data.is_cuda:
            # tell the caching allocator this storage is read on the current stream
            weight.data.record_stream(torch.cuda.current_stream())
        buf.copy_(weight.data, non_blocking=copy_non_blocking)
        offloaded.append(weight)
    return offloaded


def _free_weights(weights: list):
    for weight in weights:
        weight.data.untyped_storage().resize_(0)


def _load_weights(layer: nn.Module, non_blocking: bool):
    for weight in _weight_params(layer):
        if weight.device.type != "cuda":
            continue
        storage = weight.data.untyped_storage()
        if storage.nbytes() != 0:
            continue
        key = id(weight)
        buf = _param_cpu_buffers.get(key)
        if buf is None:
            raise RuntimeError("block swap: offloaded weight has no CPU buffer to restore from")
        pinned = _param_cpu_buffers_pinned.get(key, False)
        storage.resize_(buf.numel() * buf.element_size())
        weight.data.copy_(buf, non_blocking=non_blocking and pinned)


def materialize_optimizer_params(optimizer) -> int:
    # refill offloaded weights so optimizer.step() sees real storage
    n = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p is None or p.grad is None or p.device.type != "cuda":
                continue
            storage = p.data.untyped_storage()
            if storage.nbytes() == 0:
                buf = _param_cpu_buffers.get(id(p))
                if buf is None:
                    continue
                storage.resize_(buf.numel() * buf.element_size())
                p.data.copy_(buf, non_blocking=False)
                n += 1
    return n


def _state_dict_materialize_hook(module: nn.Module, state_dict, prefix, local_metadata):
    # offloaded weights have empty GPU storage; substitute their CPU buffer contents
    for name, p in module.named_parameters():
        if p.device.type == "cuda" and p.data.untyped_storage().nbytes() == 0:
            buf = _param_cpu_buffers.get(id(p))
            key = prefix + name
            if buf is not None and key in state_dict:
                state_dict[key] = buf.detach().clone()
    return state_dict


def _load_state_dict_materialize_pre_hook(
    module: nn.Module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
):
    # re-allocate empty storages so load_state_dict can copy into them
    for _, p in module.named_parameters():
        if p.device.type == "cuda":
            storage = p.data.untyped_storage()
            if storage.nbytes() == 0:
                storage.resize_(p.numel() * p.element_size())


def _make_load_state_dict_reoffload_post_hook():
    # re-offload immediately after this block's weights load
    def hook(module: nn.Module, incompatible_keys):
        offloaded = _offload_weights(module, non_blocking=False)
        _free_weights(offloaded)

    return hook


def swap_weight_devices_cuda(
    device: torch.device,
    layer_to_cpu: nn.Module,
    layer_to_cuda: nn.Module,
    stream_out: Optional[torch.Stream] = None,
    stream_in: Optional[torch.Stream] = None,
):
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

    stream_out = stream_out or torch.Stream(device="cuda")  # reuse caller's streams to avoid per-call stream creation
    stream_in = stream_in or torch.Stream(device="cuda")
    with torch.cuda.stream(stream_out):
        offloaded = _offload_weights(layer_to_cpu, non_blocking=True)
    with torch.cuda.stream(stream_in):
        _load_weights(layer_to_cuda, non_blocking=True)  # allocates before layer_to_cpu is freed, needs +1 block headroom

    stream_out.synchronize()  # D2H must land before its storage is freed
    _free_weights(offloaded)
    stream_in.synchronize()

    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value


def swap_weight_devices_no_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    """
    not tested
    """
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs: list[Tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]] = []
    for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
        if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
            weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

    # device to cpu
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

    _synchronize_device(device)

    # cpu to device
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
        module_to_cuda.weight.data = cuda_data_view

    _synchronize_device(device)


def weighs_to_device(layer: nn.Module, device: torch.device):
    if device.type == "cuda":
        _load_weights(layer, non_blocking=False)
    else:
        offloaded = _offload_weights(layer, non_blocking=False)
        _free_weights(offloaded)


class Offloader:
    """
    common offloading class
    """

    def __init__(self, num_blocks: int, blocks_to_swap: int, device: torch.device, debug: bool = False):
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.debug = debug

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.cuda_available = device.type == "cuda"
        # created once and reused across swaps, avoids a stream-create call on every block swap
        self._swap_stream_out = torch.Stream(device="cuda") if self.cuda_available else None
        self._swap_stream_in = torch.Stream(device="cuda") if self.cuda_available else None

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        if self.cuda_available:
            swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda, self._swap_stream_out, self._swap_stream_in)
        else:
            swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)

    @disable_compiler
    def _submit_move_blocks(self, blocks, block_idx_to_cpu, block_idx_to_cuda):
        def move_blocks(bidx_to_cpu, block_to_cpu, bidx_to_cuda, block_to_cuda):
            if self.debug:
                start_time = time.perf_counter()
                print(f"Move block {bidx_to_cpu} to CPU and block {bidx_to_cuda} to {'CUDA' if self.cuda_available else 'device'}")

            self.swap_weight_devices(block_to_cpu, block_to_cuda)

            if self.debug:
                print(f"Moved blocks {bidx_to_cpu} and {bidx_to_cuda} in {time.perf_counter() - start_time:.2f}s")
            return bidx_to_cpu, bidx_to_cuda  # , event

        block_to_cpu = blocks[block_idx_to_cpu]
        block_to_cuda = blocks[block_idx_to_cuda]

        self.futures[block_idx_to_cuda] = self.thread_pool.submit(
            move_blocks, block_idx_to_cpu, block_to_cpu, block_idx_to_cuda, block_to_cuda
        )

    @disable_compiler
    def _wait_blocks_move(self, block_idx):
        if block_idx not in self.futures:
            return

        if self.debug:
            print(f"Wait for block {block_idx}")
            start_time = time.perf_counter()

        future = self.futures.pop(block_idx)
        _, bidx_to_cuda = future.result()

        assert block_idx == bidx_to_cuda, f"Block index mismatch: {block_idx} != {bidx_to_cuda}"

        if self.debug:
            print(f"Waited for block {block_idx}: {time.perf_counter() - start_time:.2f}s")


# Gradient tensors
_grad_t = Union[tuple[torch.Tensor, ...], torch.Tensor]


class ModelOffloader(Offloader):
    """
    supports forward offloading
    """

    def __init__(
        self,
        blocks: Union[list[nn.Module], nn.ModuleList],
        blocks_to_swap: int,
        device: torch.device,
        supports_backward: bool = True,
        debug: bool = False,
    ):
        super().__init__(len(blocks), blocks_to_swap, device, debug)

        self.supports_backward = supports_backward
        self.forward_only = not supports_backward  # forward only offloading: can be changed to True for inference

        # keep state_dict/load_state_dict correct while weights are offloaded
        num_blocks = len(blocks)
        for i, block in enumerate(blocks):
            block._register_state_dict_hook(_state_dict_materialize_hook)
            block._register_load_state_dict_pre_hook(_load_state_dict_materialize_pre_hook, with_module=True)
            if i >= num_blocks - blocks_to_swap:
                block.register_load_state_dict_post_hook(_make_load_state_dict_reoffload_post_hook())

        if self.supports_backward:
            # register backward hooks
            self.remove_handles = []
            for i, block in enumerate(blocks):
                hook = self.create_backward_hook(blocks, i)
                if hook is not None:
                    handle = block.register_full_backward_hook(hook)
                    self.remove_handles.append(handle)

    def set_forward_only(self, forward_only: bool):
        self.forward_only = forward_only

    def __del__(self):
        if self.supports_backward:
            for handle in self.remove_handles:
                handle.remove()

    def create_backward_hook(
        self, blocks: Union[list[nn.Module], nn.ModuleList], block_index: int
    ) -> Optional[Callable[[nn.Module, _grad_t, _grad_t], Union[None, _grad_t]]]:
        # -1 for 0-based index
        num_blocks_propagated = self.num_blocks - block_index - 1
        swapping = num_blocks_propagated > 0 and num_blocks_propagated <= self.blocks_to_swap
        waiting = block_index > 0 and block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        # create  hook
        block_idx_to_cpu = self.num_blocks - num_blocks_propagated
        block_idx_to_cuda = self.blocks_to_swap - num_blocks_propagated
        block_idx_to_wait = block_index - 1

        def backward_hook(module: nn.Module, grad_input: _grad_t, grad_output: _grad_t):
            if self.debug:
                print(f"Backward hook for block {block_index}")

            if swapping:
                self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            if waiting:
                self._wait_blocks_move(block_idx_to_wait)
            return None

        return backward_hook

    def prepare_block_devices_before_forward(self, blocks: Union[list[nn.Module], nn.ModuleList]):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if self.debug:
            print(f"Prepare block devices before forward")

        for b in blocks[0 : self.num_blocks - self.blocks_to_swap]:
            b.to(self.device)
            weighs_to_device(b, self.device)  # make sure weights are on device

        for b in blocks[self.num_blocks - self.blocks_to_swap :]:
            b.to(self.device)  # move block to device first. this makes sure that buffers (non weights) are on the device
            weighs_to_device(b, torch.device("cpu"))  # make sure weights are on cpu

        _synchronize_device(self.device)
        _clean_memory_on_device(self.device)

    def wait_for_block(self, block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self._wait_blocks_move(block_idx)

    def submit_move_blocks(self, blocks: Union[list[nn.Module], nn.ModuleList], block_idx: int):
        # check if blocks_to_swap is enabled
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        # if backward is enabled, we do not swap blocks in forward pass more than blocks_to_swap, because it should be on GPU
        if not self.forward_only and block_idx >= self.blocks_to_swap:
            return

        block_idx_to_cpu = block_idx
        block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
        # this works for forward-only offloading. move upstream blocks to cuda
        block_idx_to_cuda = block_idx_to_cuda % self.num_blocks
        self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)


# endregion

# region cpu offload utils


def to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, list):
        return [to_device(elem, device) for elem in x]
    elif isinstance(x, tuple):
        return tuple(to_device(elem, device) for elem in x)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        return x


def to_cpu(x: Any) -> Any:
    """
    Recursively moves torch.Tensor objects (and containers thereof) to CPU.

    Args:
        x: A torch.Tensor, or a (possibly nested) list, tuple, or dict containing tensors.

    Returns:
        The same structure as x, with all torch.Tensor objects moved to CPU.
        Non-tensor objects are returned unchanged.
    """
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, list):
        return [to_cpu(elem) for elem in x]
    elif isinstance(x, tuple):
        return tuple(to_cpu(elem) for elem in x)
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    else:
        return x


def create_cpu_offloading_wrapper(func: Callable, device: torch.device) -> Callable:
    """
    Create a wrapper function that offloads inputs to CPU before calling the original function
    and moves outputs back to the specified device.

    Args:
        func: The original function to wrap.
        device: The device to move outputs back to.

    Returns:
        A wrapped function that offloads inputs to CPU and moves outputs back to the specified device.
    """

    def wrapper(orig_func: Callable) -> Callable:
        def custom_forward(*inputs):
            nonlocal device, orig_func
            cuda_inputs = to_device(inputs, device)
            outputs = orig_func(*cuda_inputs)
            return to_cpu(outputs)

        return custom_forward

    return wrapper(func)


# endregion
