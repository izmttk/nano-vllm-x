from core.executor import Executor
import time
import multiprocessing as mp

from utils import bind_parent_process_lifecycle

@bind_parent_process_lifecycle
def executor_process(tp_size, pp_size, device_ids, nccl_port):
    executor = Executor(
        tp_size=tp_size,
        pp_size=pp_size,
        device_ids=device_ids,
        nccl_port=nccl_port,
    )
    
    print(f"Executor process started with TP size {tp_size}, PP size {pp_size}, and device IDs {device_ids}.")
    
    while True:
        time.sleep(1)

if __name__ == "__main__":
    tp_size = 2
    pp_size = 1
    device_ids = [0, 1]
    
    nccl_port = 44444
    
    executor_process = mp.Process(
        target=executor_process,
        args=(tp_size, pp_size, device_ids, nccl_port),
        name="executor_process"
    )
    executor_process.start()
    print("Executor process started.")
    
    
    while True:
        time.sleep(1)
