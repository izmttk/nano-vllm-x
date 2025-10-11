import os
from core.worker_client import WorkerClient

import threading
from concurrent.futures import Future
import torch

from core.common import ForwardBatch

class Executor:
    def __init__(
        self,
        model: str,
        kv_cache_size: int,
        tp_size: int,
        pp_size: int,
        nccl_port: int = 29500,
        device_ids: list[int] | None = None,
    ):
        self.kv_cache_size = kv_cache_size
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.nccl_port = nccl_port
        self.device_ids = device_ids
        
        assert device_ids is None or len(device_ids) == tp_size * pp_size , \
            "device_ids should have the same length as tp_size * pp_size"
        if device_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))

        self.workers: list[WorkerClient] = []
        self.driver_worker: WorkerClient
        for pp_rank in range(pp_size):
            for tp_rank in range(tp_size):
                is_driver_worker = False
                if tp_rank == 0 and pp_rank == pp_size - 1:
                    is_driver_worker = True
                worker = WorkerClient(
                    model=model,
                    kv_cache_size=kv_cache_size,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    pp_rank=pp_rank,
                    pp_size=pp_size,
                    nccl_port=self.nccl_port,
                    is_driver_worker=is_driver_worker
                )
                if is_driver_worker:
                    self.driver_worker = worker
                self.workers.append(worker)

        self.collect_response_thread = threading.Thread(
            target=self.collect_response
        )
        self.collect_response_thread.start()
        self.pending = {}  # 跟踪进行中的请求 {request_id: future}

    def collect_response(self):
        while True:
            request_id, data = self.driver_worker.recv_response()
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
        print("Executor stopped response collection.")

    def shutdown(self):
        self.execute("shutdown")

        for worker in self.workers:
            worker.worker_process.join()
        self.collect_response_thread.join()

        print("Executor has been shut down.")

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
        for worker in self.workers:
            worker.send_request(request_id, request)
        return future.result()

    def execute_model(self, batch: ForwardBatch):
        return self.execute("execute_model", batch=batch)
