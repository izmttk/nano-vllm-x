import torch
import time
from distributed.parallel_state import (
    get_world_group,
    get_tp_group,
    get_pp_group,
    init_distributed_environment,
    initialize_model_parallel,
    destroy_distributed_environment,
    destroy_model_parallel
)
from core.model_runner import ModelRunner

class Worker:
    def __init__(
        self,
        tp_rank: int,
        tp_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
    ):
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.nccl_port = nccl_port
        
        self.rank = pp_rank * tp_size + tp_rank
        self.world_size = pp_size * tp_size


    def init_environment(self):
        init_distributed_environment(
            word_size=self.world_size,
            rank=self.rank,
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{self.nccl_port}",
        )
        initialize_model_parallel(
            self.tp_size,
            self.pp_size,
            self.tp_rank,
            self.pp_rank,
        )
        
        self.tp_group = get_tp_group()
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        self.device = torch.device(f"cuda:{self.rank}")
        self.model_runner = ModelRunner(
            model="/opt/models/Qwen3-8B",
            rank=self.rank,
            device=self.device,
        )
        print(f"Worker {self.rank} started with TP rank {self.tp_rank}, PP rank {self.pp_rank}.")
        
    def destroy_environment(self):
        destroy_model_parallel()
        destroy_distributed_environment()
        print(f"Worker {self.rank} destroyed its environment.")
        
    def load_model(self):
        self.model_runner.load_model()

    
    def execute_model(self):
        print(f"Worker {self.rank} is executing the model.")
        time.sleep(3)  # Simulate some processing time
        return f"WORKER {self.rank} RESULT"