import torch
import torch.distributed as dist

import os
import multiprocessing as mp
import sys
import psutil
import signal
from utils import kill_process_tree, kill_itself_when_parent_died
from communication_op import broadcast_tensor_dict
from parallel_state import (
    get_tp_group,
    get_pp_group,
    init_distributed_environment,
    initialize_model_parallel,
    destroy_distributed_environment,
    destroy_model_parallel
)

def run_worker(
    tp_rank: int,
    pp_rank: int,
    tp_size: int,
    pp_size: int,
    backend: str = "nccl",
    init_method: str = "env://"
):
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    world_size = tp_size * pp_size
    rank = pp_rank * tp_size + tp_rank

    try:
        init_distributed_environment(
            word_size=world_size,
            rank=rank,
            backend=backend,
            init_method=init_method,
        )
        initialize_model_parallel(tp_size, pp_size, tp_rank, pp_rank)

        # ======= TEST BEGINS =======

        print(f"Rank {rank}: Allocated {torch.cuda.memory_allocated(rank)/1e9:.2f} GB, "
            f"Reserved {torch.cuda.memory_reserved(rank)/1e9:.2f} GB")
        print(f"Worker {rank} started with TP rank {tp_rank}, PP rank {pp_rank}.")
        print(get_tp_group().rank(), get_tp_group().size(), dist.get_rank(), dist.get_rank(get_tp_group()), dist.get_process_group_ranks(get_tp_group()))

        # ======= TEST torch gather_object =======
        local_state = {
            "tp_rank": tp_rank,
            "pp_rank": pp_rank,
            "world_size": world_size,
            "rank": rank,
            "device": torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
        }

        all_states = [None] * get_tp_group().size()
        dist.all_gather_object(all_states, local_state, group=get_tp_group())
        print(f"Worker {rank} gathered states: {all_states} in TP.")


        all_states = [None] * get_pp_group().size()
        dist.all_gather_object(all_states, local_state, group=get_pp_group())
        print(f"Worker {rank} gathered states: {all_states} in PP.")

        # ======== TEST broadcast_tensor_dict ========
        if tp_rank == 0:
            tensor_dict = {
                "tensor1": torch.randn(3,4).cuda(),
                "tensor2": torch.randn(4,3).cuda(),
            }
        else:
            tensor_dict = None

        tensor_dict = broadcast_tensor_dict(tensor_dict, 0)

        print(f"tp_rank: {tp_rank}, pp_rank: {pp_rank}, rank: {rank}, device: {torch.cuda.current_device()}, tensor_dict: {tensor_dict}")

        # ======== TEST ENDS ========

        destroy_model_parallel()
        destroy_distributed_environment()

    except Exception as e:
        print(f"Worker {rank} encountered an error: {e}")
        parent_process.send_signal(signal.SIGQUIT)


def lanuch_processes(
    tp_size: int = 1,
    pp_size: int = 1,
    backend: str = "nccl",
    init_method: str = "env://",
    device_ids: list[int] = None,
):
    processes: list[mp.Process] = []
    mp.set_start_method("spawn", force=True)

    assert device_ids is None or len(device_ids) == tp_size * pp_size , \
        "device_ids should have the same length as tp_size * pp_size"

    if device_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))

    for pp_rank in range(pp_size):
        for tp_rank in range(tp_size):
            p = mp.Process(
                target=run_worker,
                args=(
                    tp_rank,
                    pp_rank,
                    tp_size,
                    pp_size,
                    backend,
                    init_method
                ),
            )
            p.start()
            processes.append(p)
    
    for p in processes:
        p.join()


if __name__ == "__main__":
    tp_size = 2
    pp_size = 2
    device_ids = [4, 5, 6, 7]
    try:
        lanuch_processes(
            tp_size=tp_size,
            pp_size=pp_size,
            backend="nccl",
            init_method="tcp://localhost:44444",
            device_ids=device_ids,
        )
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        kill_process_tree(include_parent=False)