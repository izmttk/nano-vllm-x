from typing import Optional
from core.engine import Engine
import torch.multiprocessing as mp
from utils import bind_parent_process_lifecycle
import threading
from concurrent.futures import Future

class EngineClient:
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
        self.model = model
        self.kv_cache_size = kv_cache_size
        self.max_bs = max_bs
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.nccl_port = nccl_port
        self.device_ids = device_ids
        
        self.methods = {}  # 用于存储注册的方法
        
        
        self.mp_ctx = mp.get_context('spawn')
        self.input_queue = self.mp_ctx.Queue()
        self.output_queue = self.mp_ctx.Queue()
        
        
        self.collect_response_thread = threading.Thread(
            target=self.collect_response
        )
        self.collect_response_thread.start()
        self.pending = {}  # 跟踪进行中的请求 {request_id: future}
        
        self.init_engine()

    def init_engine(self):
        self.engine_process = self.mp_ctx.Process(
            target=self.engine_main_loop,
            name=f"engine",
        )
        self.engine_process.start()

    def send_request(self, request_id: str, data: dict):
        self.input_queue.put((request_id, data))

    def recv_response(self, timeout: Optional[float] = None):
        return self.output_queue.get(timeout=timeout)

    @bind_parent_process_lifecycle
    def engine_main_loop(self):
        engine = Engine(
            model=self.model,
            kv_cache_size=self.kv_cache_size,
            max_bs=self.max_bs,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            nccl_port=self.nccl_port,
            device_ids=self.device_ids,
        )
        
        self.methods["add_sequence"] = engine.add_sequence
        self.methods["step"] = engine.step
        
        while True:
            request_id, data = self.input_queue.get()  # 等待输入
            method_name = data.get('method')
            if method_name == "shutdown":
                self.output_queue.put((request_id, "shutdown"))
                break
            response = self.handle_request(data)  # 处理请求
            self.output_queue.put((request_id, response))
        engine.shutdown()
        print(f"Engine has shut down.")

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

    def collect_response(self):
        while True:
            request_id, data = self.recv_response()
            future = self.pending.pop(request_id, None)
            if future:
                if data == "shutdown":
                    future.set_result(None)
                    break
                assert isinstance(data, dict)
                if data['status'] == 'success':
                    future.set_result(data['result'])
                else:
                    future.set_exception(Exception(data['error']))
        print("Engine stopped response collection.")

    def execute(self, method, *args, **kwargs):
        """发起RPC调用，返回调用结果"""
        
        request_id = str(id(kwargs))  # 生成唯一ID
        request = {
            'method': method,
            'args': args,
            'kwargs': kwargs
        }
        
        future = Future()
        self.pending[request_id] = future
        print(f"EngineClient sending request {request_id} for method {method} with args {args} and kwargs {kwargs}")
        self.send_request(request_id, request)
        return future.result()

    def shutdown(self):
        self.execute("shutdown")

        self.engine_process.join()
        self.collect_response_thread.join()

        print("Engine has been shut down.")
        

    def add_sequence(
        self,
        prompt_token_ids: list[int],
        sampling_params,
    ):
        self.execute(
            "add_sequence",
            prompt_token_ids,
            sampling_params,
        )
    
    def step(self) -> list[int]:
        output_ids = self.execute("step")
        return output_ids