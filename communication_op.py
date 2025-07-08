import torch
import torch.distributed as dist

from parallel_state import get_tp_group

from typing import Any, Callable, Optional, Tuple, Union

def tensor_model_parallel_all_reduce(input: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    tp_group = get_tp_group()
    dist.all_reduce(input, group=tp_group)
    return input

def tensor_model_parallel_all_gather(input: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""

    if dim < 0:
        # Convert negative dim to positive.
       dim += input.dim()
    tp_group = get_tp_group()

    input_size = input.size()
    world_size = tp_group.size()
    # NOTE: we have to use concat-style all-gather here,
    # stack-style all-gather has compatibility issues with
    # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
    output_size = (input_size[0] * world_size, *input_size[1:])
    # Allocate output tensor.
    output_tensor = torch.empty(output_size, dtype=input.dtype, device=input.device)
    # All-gather.
    dist.all_gather_into_tensor(output_tensor, input, group=tp_group)
    # Reshape
    output_tensor = output_tensor.reshape((world_size, ) + input_size)
    output_tensor = output_tensor.movedim(0, dim)
    output_tensor = output_tensor.reshape(input_size[:dim] +
                                            (world_size *
                                            input_size[dim], ) +
                                            input_size[dim + 1:])
    return output_tensor

def tensor_model_parallel_reduce_scatter(input: torch.Tensor,
                                         dim: int = -1) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""

    tp_group = get_tp_group()
    world_size =  tp_group.size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input
    assert -input.dim() <= dim < input.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input.size()}")

    if dim < 0:
        # Convert negative dim to positive.
        dim += input.dim()

    # Note: This will produce an incorrect answer if we don't make
    # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
    input_tensor = input.movedim(0, dim).contiguous()

    assert input_tensor.shape[0] % world_size == 0
    chunk_size = input_tensor.shape[0] // world_size
    output_shape = (chunk_size, ) + input_tensor.shape[1:]

    output_tensor = torch.empty(output_shape,
                                dtype=input_tensor.dtype,
                                device=input_tensor.device)

    # Perform reduce-scatter operation
    dist.reduce_scatter_tensor(output_tensor,
                                input_tensor,
                                group=tp_group)

    # Reshape before returning
    return output_tensor.movedim(0, dim).contiguous()


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group.
    NOTE: We assume that the input tensor is on the same device across
    all the ranks.
    NOTE: `dst` is the local rank of the destination rank.
    """

    tp_group = get_tp_group()
    world_size = tp_group.size()
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()

    # Allocate output tensor.
    rank_in_group = tp_group.rank()
    if rank_in_group == dst:
        gather_list = [torch.empty_like(input_) for _ in range(world_size)]
    else:
        gather_list = None
    # Gather.
    dist.gather(input_,
                gather_list,
                group_dst=dst,
                group=tp_group)
    if rank_in_group == dst:
        output_tensor = torch.cat(gather_list, dim=dim)
    else:
        output_tensor = None
    return output_tensor

def broadcast_tensor_dict(tensor_dict: Optional[dict[Any, Union[torch.Tensor, Any]]] = None,
                          src: int = 0):
    """Broadcast the input tensor dictionary.
    NOTE: `src` is the local rank of the source rank.
    """
    tp_group = get_tp_group()
    world_size = tp_group.size()
    # Bypass the function if we are using only 1 GPU.
    if (not dist.is_initialized() or world_size == 1):
        return tensor_dict

    assert src < world_size, f"Invalid src rank ({src})"

    rank_in_group = tp_group.rank()

    if rank_in_group == src:
        metalist = []
        for key, tensor in tensor_dict.items():
            metalist.append((key, tensor.device.type, tensor.dtype, tensor.shape))
    else:
        metalist = None

    # 先广播 metalist
    recv = [metalist]
    dist.broadcast_object_list(recv, group_src=src, group=tp_group)
    metalist = recv[0]

    result = {}
    for key, device, dtype, shape in metalist:
        if rank_in_group == src:
            tensor = tensor_dict[key]
        else:
            tensor = torch.empty(shape, dtype=dtype, device=device)
        dist.broadcast(tensor, group_src=src)  # 直接张量广播
        result[key] = tensor
    return result