import torch
from torch.distributed import ProcessGroup
import torch.distributed as dist
import datetime
from typing import Optional

WORLD: ProcessGroup | None = None
TP: ProcessGroup | None = None
PP: ProcessGroup | None = None

# Attention: 所有的 rank 都指的是 global rank，除非变量或参数命名为 rank_in_group, tp_rank, pp_rank

def get_world_group() -> ProcessGroup:
    assert WORLD is not None, "Distributed process group is not initialized."
    return WORLD

def get_tp_group() -> ProcessGroup:
    """Get the tensor parallel process group."""
    assert TP is not None, "Tensor parallel process group is not initialized."
    return TP

def get_pp_group() -> ProcessGroup:
    """Get the pipeline parallel process group."""
    assert PP is not None, "Pipeline parallel process group is not initialized."
    return PP

def initialize_model_parallel(tp_size: int, pp_size: int, tp_rank: int, pp_rank: int):
    global TP, PP

    tp_group_ranks = [pp_rank * tp_size + k for k in range(tp_size)]
    pp_group_ranks = [tp_rank + k * tp_size for k in range(pp_size)]

    TP = dist.new_group(ranks=tp_group_ranks)
    PP = dist.new_group(ranks=pp_group_ranks)

def destroy_model_parallel():
    global TP, PP
    if TP is not None:
        dist.destroy_process_group(TP)
    if  PP is not None:
        dist.destroy_process_group(PP)
    TP = None
    PP = None

def init_distributed_environment(
    word_size: int = -1,
    rank: int = -1,
    backend: str = "nccl",
    init_method: str = "env://",
):
    global WORLD

    if dist.is_initialized():
        print("Distributed process group is already initialized.")
        return
    # os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ["NCCL_IB_DISABLE"] = "1"        # 禁用InfiniBand
    # os.environ["NCCL_P2P_DISABLE"] = "1"       # 禁用P2P通信
    # os.environ["NCCL_SHM_DISABLE"] = "1"       # 禁用共享内存
    # os.environ["NCCL_SOCKET_IFNAME"] = "eth0"  # 使用指定网络接口
    # os.environ["NCCL_PORT_RANGE"] = "50000-50100"  # 限制端口范围

    if backend == "nccl":
        # Set the device for each process based on its rank
        torch.cuda.set_device(rank)

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=word_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=30),  # Set a timeout for the process group
        device_id=torch.device("cuda", rank) if backend == "nccl" else None,
    )

    WORLD = dist.group.WORLD

def destroy_distributed_environment():
    global WORLD
    if dist.is_initialized():
        dist.destroy_process_group()
    WORLD = None

def is_initialized() -> bool:
    return dist.is_initialized()

def get_first_rank(group: Optional[ProcessGroup] = None) -> int:
    if group is None:
        group = dist.group.WORLD
    assert group is not None
    group_ranks = dist.get_process_group_ranks(group)
    return group_ranks[0]

def get_last_rank(group: Optional[ProcessGroup] = None) -> int:
    if group is None:
        group = dist.group.WORLD
    assert group is not None
    group_ranks = dist.get_process_group_ranks(group)
    return group_ranks[-1]

def is_first_rank(group: Optional[ProcessGroup] = None) -> bool:
    if group is None:
        group = dist.group.WORLD
    assert group is not None
    rank = dist.get_rank() # get global rank
    return rank == get_first_rank(group)

def is_last_rank(group: Optional[ProcessGroup] = None) -> bool:
    if group is None:
        group = dist.group.WORLD
    assert group is not None
    rank = dist.get_rank() # get global rank
    return rank == get_last_rank(group)

def prev_rank(group: Optional[ProcessGroup] = None) -> int:
    if group is None:
        group = dist.group.WORLD
    assert group is not None
    rank = dist.get_rank(group)
    group_ranks = dist.get_process_group_ranks(group)
    return group_ranks[rank - 1]

def next_rank(group: Optional[ProcessGroup] = None) -> int:
    if group is None:
        group = dist.group.WORLD
    assert group is not None
    rank = dist.get_rank(group)
    group_ranks = dist.get_process_group_ranks(group)
    return group_ranks[rank + 1]