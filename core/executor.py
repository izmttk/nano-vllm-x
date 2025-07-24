import os
from core.worker_client import WorkerClient
class Executor:
    def __init__(
        self,
        tp_size: int,
        pp_size: int,
        nccl_port: int = 29500,
        device_ids: list[int] | None = None,
    ):
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.nccl_port = nccl_port
        self.device_ids = device_ids
        
        
        assert device_ids is None or len(device_ids) == tp_size * pp_size , \
            "device_ids should have the same length as tp_size * pp_size"
        if device_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))
            
        self.workers: list[WorkerClient] = []
            
        for pp_rank in range(pp_size):
            for tp_rank in range(tp_size):
                worker = WorkerClient(
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    pp_rank=pp_rank,
                    pp_size=pp_size,
                    nccl_port=self.nccl_port,
                )
                self.workers.append(worker)
    
    
    def execute_model(self):
        pass