import os
import time
from random import randint, seed
from core.llm import LLM
from core.common import SamplingParams
import asyncio

async def consume_async_gen(async_gen):
    num = 0
    async for _ in async_gen:
        num += 1
    return num

async def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = "../Qwen3-0.6B"
    llm = LLM(
        model=path,
        gpu_memory_utilization=0.6,
        max_bs=50,
        tp_size=1,
        pp_size=1,
        nccl_port=29500,
        device_ids=[0],
    )

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    await consume_async_gen(llm.generate("Benchmark: ", SamplingParams(max_tokens=10)))
    print("Start Profiling ...")
    t = time.time()
    
    tasks = []
    for p, s in zip(prompt_token_ids, sampling_params):
        tasks.append(consume_async_gen(llm.generate(p, s)))
    nums = await asyncio.gather(*tasks)
    
    t = (time.time() - t)
    total_tokens = sum(s.max_tokens or 0 for s in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
    llm.shutdown()


if __name__ == "__main__":
    asyncio.run(main())