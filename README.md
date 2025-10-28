# Nano-vLLM-X

**Nano vLLM** with Radi**X**-Tree based Cache, and more.

该项目受 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm/tree/main) 启发，提供一个从零开始构建的 LLM 推理框架

程序的架构遵循了 [vLLM](https://github.com/vllm-project/vllm) v1 相似的组织安排，但是与 vLLM 不同的是，该项目的 KV 缓存系统使用的是 [SGLang](https://github.com/sgl-project/sglang) 的 Radix Cache 实现。

## Features

- 轻量但完整的代码实现
- 持续批处理（Continuous Batching）
- OpenAI 兼容的 API
- 基于 Radix Tree 的 Prefix Caching
- 张量并行（Tensor Parallelism）
- 流水线并行（Pipeline Parallelism）
- CUDA Graph 支持（仅 Decoding 阶段）

## Requirements

```plaintext
torch >= 2.8.0
transformers >= 4.51.0
fastapi >= 0.95.0
flashinfer-python >= 0.2.0
psutil
```

NOTE: 我发现当 nccl 版本 < 2.27.3 时，分布式环境的销毁会存在一些问题，建议升级 nvidia-nccl-cu12 至 2.27.3 及以上版本，torch 2.8.0 的依赖中已经包含该版本的 nccl。具体原因需要进一步调查。

## Quick Start

启动 API 服务

```bash
python3 -m entrypoints.openai.api --model <model_path> --host 0.0.0.0 --port 8000
```

Offline Inference

```py
from core.llm import LLM

async main(prompt: str, *args, **kwargs):
    llm = LLM(*args, **kwargs)
    async for token in llm.generate(
        prompt,
        SamplingParams(
            max_new_tokens=50,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )
    ):
        print(token, end="", flush=True)
```

## Benchmarks

Experiment Environment:

- GPU: A100 40GB
- Model: Qwen3-0.6B
- Number of Requests: 256
- Prompt Length: random 100 ~ 1024
- Generation Length: random 100 ~ 1024
- Script: [bench.py](bench.py)

Results:

| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|------------------|---------------|----------|-----------------------|
| vLLM v0.11.0     | 133966        | 18.24    |  7343.96              |
| ours             | 133966        | 14.83    |  9032.37              |

## TODO

- Graceful Shutdown
- Better Logging System
- Benchmark Metrics on API Server
- More Configurable Options

[WIP]
