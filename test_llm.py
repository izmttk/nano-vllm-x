import asyncio
from core.llm import LLM, SamplingParams

async def main():
    print("Initializing LLM...")
    llm = LLM(
        model="../Qwen3-0.6B",
        gpu_memory_utilization=0.9,
        max_bs=4,
        tp_size=1,
        pp_size=1,
        nccl_port=29500,
        device_ids=[0],
    )
    prompt = "Hello, my name is"
    print(prompt, end='', flush=True)
    
    async for token in llm.generate(
        prompt,
        SamplingParams(
            max_new_tokens=50,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        ),
    ):
        print(token, end='', flush=True)
    print()

    llm.shutdown()

if __name__ == "__main__":
    print("Starting async main...")
    asyncio.run(main())