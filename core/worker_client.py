from core.worker import Worker
import multiprocessing as mp
import time
from utils import bind_parent_process_lifecycle

class WorkerClient:
    def __init__(
        self,
        tp_rank: int,
        tp_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int = 29500
    ):
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.rank = pp_rank * tp_size + tp_rank
        self.world_size = pp_size * tp_size

        self.nccl_port = nccl_port
        self.init_worker()
        
    def init_worker(self):
        # reader, writer = mp.Pipe(duplex=False)
        self.worker_process = mp.Process(
            target=self.worker_main_loop,
            name=f"worker-{self.rank}",
        )
        self.worker_process.start()
        
    @bind_parent_process_lifecycle
    def worker_main_loop(self):
        worker = Worker(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            nccl_port=self.nccl_port
        )
        worker.init_environment()
        worker.load_model()
        while True:
            time.sleep(1)
