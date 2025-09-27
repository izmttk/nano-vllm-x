from typing import Optional

import torch
from core.common import SamplingParams, Sequence, ForwardMode
from core.executor import Executor
from core.scheduler import Scheduler


class Engine:
    """
    The Engine class coordinates the scheduler and the executor to run the LLM inference.
    """

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
        self.model_executor = Executor(
            model=model,
            kv_cache_size=kv_cache_size,
            tp_size=tp_size,
            pp_size=pp_size,
            nccl_port=nccl_port,
            device_ids=device_ids,
        )
        self.scheduler = Scheduler(
            kv_cache_size=kv_cache_size,
            max_bs=max_bs
        )

    def add_sequence(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
    ):
        """
        Add a new sequence to the engine's scheduler.
        """
        seq = Sequence(
            token_ids=prompt_token_ids,
            num_tokens=len(prompt_token_ids),
            prompt_len=len(prompt_token_ids),
            sampling_params=sampling_params,
        )
        self.scheduler.add_sequence(seq)

    def step(self) -> list[int]:
        """
        Performs one step of inference.

        1. Schedules a batch of sequences.
        3. Executes the model.
        5. Updates the sequences.
        """
        batch = self.scheduler.schedule()
        if not batch:
            return []

        output_ids = self.model_executor.execute_model(batch)
        
        # Update sequences with the model output
        for seq, new_token_id in zip(batch.seqs, output_ids):
            self.scheduler.update_sequence(seq, new_token_id)

            if self._is_sequence_finished(seq):
                self.scheduler.finish_sequence(seq)
        
        return output_ids

    def _is_sequence_finished(self, seq: Sequence) -> bool:
        # Check for stop tokens
        if seq.token_ids[-1] == seq.sampling_params.eos_token_id and not seq.sampling_params.ignore_eos:
            return True
        # Check for max tokens
        if seq.num_tokens >= seq.prompt_len + seq.sampling_params.max_new_tokens:
            return True
        return False

    def shutdown(self):
        self.model_executor.shutdown()
