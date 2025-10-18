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
        gpu_memory_utilization: float = 0.9,
        max_bs: int = 50,
        tp_size: int = 1,
        pp_size: int = 1,
        nccl_port: int = 29500,
        device_ids: list[int] | None = None,
        enforce_eager: bool = False,
    ):
        self.engine = EngineClient(
            model,
            gpu_memory_utilization,
            max_bs,
            tp_size,
            pp_size,
            nccl_port,
            device_ids,
            enforce_eager,
        )
    
        self.tokenizer = init_tokenizer(model)
        self.event_loop = asyncio.get_event_loop()

        self.request_states: dict[int, asyncio.Queue[str | None]] = {}
        self.process_outputs_task = asyncio.create_task(self._process_outputs_async())

    async def _process_outputs_async(self):
        while True:
            outputs = await self.event_loop.run_in_executor(None, self.engine.get_output)
            if outputs is None: # Shutdown signal
                break
            for output in outputs:
                seq_id = output.seq_id
                new_token_id = output.new_token_id
                if seq_id in self.request_states:
                    q = self.request_states[seq_id]
                    token_str = self.detokenize([[new_token_id]])[0]
                    if output.is_finished:
                        q.put_nowait(token_str)
                        q.put_nowait(None)  # Sentinel for end of generation
                        del self.request_states[seq_id]
                    else:
                        q.put_nowait(token_str)

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        return self.tokenizer(texts)["input_ids"] # type: ignore

    def detokenize(self, token_ids: list[list[int]]) -> list[str]:
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    
    async def ready(self):
        pass
    
    async def generate(
        self,
        prompts: str | list[int],
        params: SamplingParams,
    ) -> AsyncGenerator[str, None]:
        seq_id = uuid.uuid4().int
        try:
            if isinstance(prompts, list):
                token_ids = prompts
            else:
                token_ids = self.tokenize([prompts])[0]
            q: asyncio.Queue[str | None] = asyncio.Queue()
            
            if params.eos_token_id == -1:
                params.eos_token_id = self.tokenizer.eos_token_id  # type: ignore
            
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
                
        # If the request is disconnected by the client, the
        # generate() task will be canceled. So, we abort the
        # request if we end up here.
        except asyncio.CancelledError:
            self.abort(seq_id)
            raise

    def abort(self, seq_id: int):
        if seq_id in self.request_states:
            self.engine.abort_sequence(seq_id)
            del self.request_states[seq_id]

    def shutdown(self):
        self.engine.shutdown()
        self.process_outputs_task.cancel()
