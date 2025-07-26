from core.executor import Executor
import time
import multiprocessing as mp
import asyncio

from utils import bind_parent_process_lifecycle

@bind_parent_process_lifecycle
def executor_process(
    tp_size: int,
    pp_size: int,
    device_ids: list[int],
    nccl_port: int
):
    executor = Executor(
        tp_size=tp_size,
        pp_size=pp_size,
        device_ids=device_ids,
        nccl_port=nccl_port,
    )
    print(f"Executor process started with TP size {tp_size}, PP size {pp_size}, and device IDs {device_ids}.")
    print(executor.execute_model())
    executor.shutdown()
    print("All finished.")

if __name__ == "__main__":
    tp_size = 2
    pp_size = 1
    device_ids = [0, 1]
    
    nccl_port = 44444
    
    process = mp.Process(
        target=executor_process,
        args=(tp_size, pp_size, device_ids, nccl_port),
        name="executor_process"
    )
    print("Executor process started.")
    process.start()

