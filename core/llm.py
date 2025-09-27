from core.engine_client import EngineClient
from core.common import SamplingParams

class AsyncLLM:
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
        
    def generate(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
    ) -> list[int]:
        """
        Add a new sequence to the engine's scheduler.
        """
        self.engine.add_sequence(prompt_token_ids, sampling_params)
        # TODO WIP
        pass