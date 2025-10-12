from core.common import SamplingParams, Sequence
from core.executor import Executor
from core.scheduler import Scheduler
from dataclasses import dataclass, field

@dataclass
class EngineOutput:
    seq_id: int
    new_token_id: int
    is_finished: bool

class Engine:
    """
    The Engine class coordinates the scheduler and the executor to run the LLM inference.
    """

    def __init__(
        self,
        model: str,
        gpu_memory_utilization: float,
        max_bs: int,
        tp_size: int,
        pp_size: int,
        nccl_port: int = 29500,
        device_ids: list[int] | None = None,
    ):
        self.model_executor = Executor(
            model=model,
            tp_size=tp_size,
            pp_size=pp_size,
            nccl_port=nccl_port,
            device_ids=device_ids,
        )
        
        kv_cache_size = self.model_executor.profile_kv_cache_size(gpu_memory_utilization)
        print(f"Max num tokens in kv cache: {kv_cache_size}")
        self.model_executor.initialize_kv_cache(kv_cache_size)

        self.scheduler = Scheduler(
            kv_cache_size=kv_cache_size,
            max_bs=max_bs
        )
        

    def add_sequence(
        self,
        sequence_id: int,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
    ):
        """
        Add a new sequence to the engine's scheduler.
        """
        seq = Sequence(
            seq_id=sequence_id,
            token_ids=prompt_token_ids,
            num_tokens=len(prompt_token_ids),
            prompt_len=len(prompt_token_ids),
            sampling_params=sampling_params,
        )
        self.scheduler.add_sequence(seq)

    def step(self) -> list[EngineOutput]:
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
        outputs: list[EngineOutput] = []
        
        # Update sequences with the model output
        for seq, new_token_id in zip(batch.seqs, output_ids):
            self.scheduler.update_sequence(seq, new_token_id)

            is_finished = self._is_sequence_finished(seq)
            if is_finished:
                self.scheduler.finish_sequence(seq)
            
            outputs.append(EngineOutput(
                seq_id=seq.seq_id,
                new_token_id=new_token_id,
                is_finished=is_finished,
            ))
        
        return outputs

    def _is_sequence_finished(self, seq: Sequence) -> bool:
        # Check for stop tokens
        if seq.token_ids[-1] == seq.sampling_params.eos_token_id and not seq.sampling_params.ignore_eos:
            return True
        # Check for max tokens
        if seq.sampling_params.max_tokens and seq.num_tokens >= seq.sampling_params.max_tokens:
            return True
        if seq.sampling_params.max_new_tokens and seq.num_tokens >= seq.prompt_len + seq.sampling_params.max_new_tokens:
            return True
        return False

    def shutdown(self):
        self.model_executor.shutdown()
        
    def has_unfinished_sequences(self):
        return self.scheduler.has_unfinished_sequences()
