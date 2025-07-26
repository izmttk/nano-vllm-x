import os
from core.worker_client import WorkerClient

import asyncio
import zmq
from zmq.asyncio import Context
import msgpack

class RpcClient:
    def __init__(
        self,
        request_endpoint: str,
        response_endpoint: str,
        num_workers: int = 1,
    ):
        self.ctx = Context.instance()
        
        self.sender = self.ctx.socket(zmq.PUB)
        self.sender.bind(request_endpoint)
        self.pending = {}  # 跟踪进行中的请求 {request_id: future}
        
        self.receiver = self.ctx.socket(zmq.PULL)
        self.receiver.bind(response_endpoint)
        
        self.num_workers = num_workers

    async def recv_ready(self):
        for i in range(self.num_workers):
            await self.receiver.recv()

    def close(self):
        """清理资源"""
        self.sender.close()
        self.receiver.close()
        self.ctx.term()

    async def start(self):
        """主循环，接收响应并处理"""
        while True:
            request_id, resp_data = await self.receiver.recv_multipart()
            await self.process_single_response(request_id, resp_data)
            
    async def process_single_response(self, request_id, resp_data):
        """处理单个响应"""
        request_id = request_id.decode()
        response = msgpack.unpackb(resp_data, raw=False)
        future = self.pending.pop(request_id, None)
        if future:
            if response['status'] == 'success':
                future.set_result(response['result'])
            else:
                future.set_exception(Exception(response['error']))

    async def execute(self, method, *args, **kwargs):
        """发起RPC调用，返回调用结果"""
        request_id = str(id(kwargs))  # 生成唯一ID
        request = {
            'method': method,
            'args': args,
            'kwargs': kwargs
        }
        packed_req = msgpack.packb(request, use_bin_type=True)
        future = asyncio.get_event_loop().create_future()
        self.pending[request_id] = future
        # 非阻塞发送
        print(f"Sending request {request_id} for method {method} with args {args} and kwargs {kwargs}")
         # 发送请求到RPC服务器
        await self.sender.send_multipart([request_id.encode(), packed_req])
        
        return await future  # 等待响应并返回结果


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

        # 注意先启动 RPC Client, 再启动 RPC Server, 确保 RPC Server 可以连接到 RPC Client
        self.rpc_client = RpcClient(
            request_endpoint=f"ipc://executor_request.ipc",
            response_endpoint=f"ipc://executor_response.ipc",
            num_workers=tp_size * pp_size,
        )
        
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

    async def startup(self):
        await self.rpc_client.recv_ready()
        self.recv_loop = asyncio.create_task(self.rpc_client.start())
        
    async def shutdown(self):
        """清理资源"""
        self.recv_loop.cancel()
        for worker in self.workers:
            worker.shutdown()

    async def execute_model(self):
        print("Executor is executing the model...")
        return await self.rpc_client.execute("execute_model")
