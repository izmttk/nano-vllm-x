from typing import Optional
import signal
import sys
from core.worker import Worker
import multiprocessing as mp
from utils import bind_parent_process_lifecycle
import queue
class WorkerClient:
    def __init__(
        self,
        tp_rank: int,
        tp_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int = 29500,
    ):
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.rank = pp_rank * tp_size + tp_rank
        self.world_size = pp_size * tp_size

        self.nccl_port = nccl_port
        
        self.methods = {}  # 用于存储注册的方法
        
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.init_worker()

    def init_worker(self):
        self.worker_process = mp.Process(
            target=self.worker_main_loop,
            name=f"worker-{self.rank}",
        )
        self.worker_process.start()
    
    def send_request(self, request_id: str, data: dict):
        self.input_queue.put((request_id, data))

    def recv_response(self, timeout: Optional[float] = None):
        return self.output_queue.get(timeout=timeout)

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
        
        self.methods["execute_model"] = worker.execute_model
        
        while True:
            request_id, data = self.input_queue.get()  # 等待输入
            method_name = data.get('method')
            if method_name == "shutdown":
                self.output_queue.put((request_id, "shutdown"))
                break
            response = self.handle_request(data)  # 处理请求
            self.output_queue.put((request_id, response))
        worker.destroy_environment()
        print(f"Worker {self.rank} has shut down.")

    def handle_request(self, request):
        """处理单个请求"""
        try:
            method_name = request['method']
            args = request.get('args', [])
            kwargs = request.get('kwargs', {})
            # 查找并调用注册的方法
            if method_name in self.methods:
                method = self.methods[method_name]
                # 执行方法
                result = method(*args, **kwargs)
                
                # 返回成功结果
                return {
                    'status': 'success',
                    'result': result
                }
            else:
                # 方法不存在
                return {
                    'status': 'error',
                    'error': f"Method '{method_name}' not found"
                }
        except Exception as e:
            # 处理异常
            return {
                'status': 'error',
                'error': str(e)
            }
