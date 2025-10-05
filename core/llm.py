from typing import AsyncGenerator
from core.engine_client import EngineClient
from core.common import SamplingParams
import asyncio
from transformers import AutoTokenizer, PreTrainedTokenizer
import uuid

def init_tokenizer(model: str) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.unk_token is None:
        tokenizer.unk_token = tokenizer.eos_token
    return tokenizer

class LLM:
    def __init__(
        self,
        model: str,
        kv_cache_size: int,
        max_bs: int,
        tp_size: int,
        pp_size: int,
        nccl_port: int = 29500,
        device_ids: list[int] | None = None,
    ):
        self.engine = EngineClient(
            model,
            kv_cache_size,
            max_bs,
            tp_size,
            pp_size,
            nccl_port,
            device_ids,
        )
    
        self.tokenizer = init_tokenizer(model)
        
        self.request_states: dict[int, asyncio.Queue[str | None]] = {}
        self.output_handler_task = asyncio.create_task(self._handle_outputs())

    
    async def _handle_outputs(self):
        while True:
            outputs = self.engine.get_output()
            for output in outputs:
                seq_id = output.seq_id
                new_token_id = output.new_token_id
                if seq_id in self.request_states:
                    q = self.request_states[seq_id]
                    token_str = self.detokenize([[new_token_id]])[0]
                    if output.is_finished:
                        q.put_nowait(None)  # Sentinel for end of generation
                        del self.request_states[seq_id]
                    else:
                        q.put_nowait(token_str)
            # 让出控制权，避免阻塞事件循环
            await asyncio.sleep(0)

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        return self.tokenizer(texts)["input_ids"] # type: ignore

    def detokenize(self, token_ids: list[list[int]]) -> list[str]:
        return self.tokenizer.batch_decode(token_ids)
    
    async def generate(
        self,
        prompts: str,
        params: SamplingParams,
    ) -> AsyncGenerator[str, None]:
        
        token_ids = self.tokenize([prompts])[0]
        seq_id = uuid.uuid4().int
        q: asyncio.Queue[str | None] = asyncio.Queue()
        
        self.request_states[seq_id] = q
        self.engine.add_sequence(
            sequence_id=seq_id,
            prompt_token_ids=token_ids,
            sampling_params=params,
        )
        
        while True:
            token_str = await q.get()
            if token_str is None:  # End of generation
                break
            yield token_str

    def shutdown(self):
        self.engine.shutdown()
        self.output_handler_task.cancel()