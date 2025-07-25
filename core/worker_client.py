from typing import Optional

from core.worker import Worker
import multiprocessing as mp
import time
from utils import bind_parent_process_lifecycle

import asyncio
import zmq
from zmq.asyncio import Context
import msgpack

class RpcServer:
    def __init__(
        self, 
        request_endpoint: str,
        response_endpoint: str,
        is_responser: bool = True
    ):
        self.ctx = Context.instance()
        self.receiver = self.ctx.socket(zmq.SUB)
        self.receiver.setsockopt(zmq.SUBSCRIBE, b"") # 订阅所有消息
        self.receiver.connect(request_endpoint)
        self.methods = {}
        
        self.sender = self.ctx.socket(zmq.PUSH)
        self.sender.connect(response_endpoint)
        
        self.is_responser = is_responser
        
    def register_method(self, name, coro_func):
        """注册可以被远程调用的方法"""
        self.methods[name] = coro_func
    
    async def handle_request(self, request_data):
        """处理单个请求"""
        try:
            request = msgpack.unpackb(request_data, raw=False)
            method_name = request['method']
            args = request.get('args', [])
            kwargs = request.get('kwargs', {})
            # 查找并调用注册的方法
            if method_name in self.methods:
                method = self.methods[method_name]
                # 执行方法（支持async和sync函数）
                if asyncio.iscoroutinefunction(method):
                    result = await method(*args, **kwargs)
                else:
                    result = method(*args, **kwargs)
                
                # 返回成功结果
                return msgpack.packb({
                    'status': 'success',
                    'result': result
                })
            else:
                # 方法不存在
                return msgpack.packb({
                    'status': 'error',
                    'error': f"Method '{method_name}' not found"
                })
        except Exception as e:
            # 处理异常
            return msgpack.packb({
                'status': 'error',
                'error': str(e)
            })

    async def ready(self):
        """发送服务就绪信号"""
        await self.sender.send(b"READY")

    async def start(self):
        """启动RPC服务"""
        while True:
            request_id, request_data = await self.receiver.recv_multipart()
            # 并行处理请求
            asyncio.create_task(
                self.process_single_request(request_id, request_data)
            )
    
    async def process_single_request(self, request_id, request_data):
        """处理单个请求并发送响应"""
        try:
            response = await self.handle_request(request_data)
            if self.is_responser:
                await self.sender.send_multipart([request_id, response])
        except Exception as e:
            if self.is_responser:
                error_msg = msgpack.dumps({
                    'status': 'error',
                    'error': f"Server error: {str(e)}"
                })
                await self.sender.send_multipart([request_id, error_msg])



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

        rpc_server = RpcServer(
            request_endpoint=f"ipc://executor_request.ipc",
            response_endpoint=f"ipc://executor_response.ipc",
            is_responser=(self.tp_rank == 0 and self.pp_rank == self.pp_size - 1)
        )
        rpc_server.register_method("execute_model", worker.execute_model)
        
        asyncio.run(rpc_server.ready())
        print(f"Worker {self.rank} is ready.")
        asyncio.run(rpc_server.start())
