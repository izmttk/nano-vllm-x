from core.executor import Executor
from transformers import AutoConfig, AutoTokenizer
import multiprocessing as mp

from utils import bind_parent_process_lifecycle

@bind_parent_process_lifecycle
def executor_process(
    tp_size: int,
    pp_size: int,
    device_ids: list[int],
    nccl_port: int
):
    
    prompt = "明月几时有，把酒问青天。不知"
    model="/opt/models/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").squeeze(0)

    executor = Executor(
        model=model,
        tp_size=tp_size,
        pp_size=pp_size,
        device_ids=device_ids,
        nccl_port=nccl_port,
    )
    print(f"Executor process started with TP size {tp_size}, PP size {pp_size}, and device IDs {device_ids}.")
    
    output_ids = executor.execute_model(input_ids)
    print(output_ids)
    print(tokenizer.decode(output_ids))
    
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

