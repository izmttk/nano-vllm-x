from typing import List, Dict, Optional
import time
import uuid
from core.executor import Executor
from core.common import Sequence, SamplingParams, SequenceStatus, ForwardBatch, ForwardMode
from core.kv_cache import KVCacheAllocator
class Scheduler:
    def __init__(self, executor: Executor, kv_cache_size: int = 1024):
        self.executor = executor
        self.sequences: Dict[int, Sequence] = {}
        self.kv_allocator = KVCacheAllocator(kv_cache_size)
        
    def add_sequence(self, 
                    prompt_token_ids: List[int], 
                    sampling_params: Optional[SamplingParams] = None) -> int:
        """添加一个新的序列到调度器"""
        if sampling_params is None:
            sampling_params = SamplingParams()
            
        seq_id = uuid.uuid4().int  # 使用UUID生成唯一的seq_id
        
        # 为序列分配KV cache空间
        required_slots = len(prompt_token_ids) + sampling_params.max_new_tokens
        allocated_slots = self.kv_allocator.alloc(required_slots)
        
        if allocated_slots is None:
            raise RuntimeError("Not enough KV cache space available")
        
        sequence = Sequence(
            seq_id=seq_id,
            status=SequenceStatus.WAITING,
            num_tokens=len(prompt_token_ids),
            token_ids=prompt_token_ids.copy(),
            sampling_params=sampling_params,
            kv_indices=allocated_slots,
            cached_kv_len=0,
        )
        
        self.sequences[seq_id] = sequence
        print(f"Added sequence {seq_id} with {len(prompt_token_ids)} prompt tokens")
        return seq_id
    
    def step(self) -> Dict[int, List[int]]:
        """执行一步生成"""
        if not self.sequences:
            return {}
        
        # 分离prefill和decode序列
        prefill_seqs = []
        decode_seqs = []
        
        for seq in self.sequences.values():
            if seq.status == SequenceStatus.FINISHED:
                continue
                
            if seq.cached_kv_len == 0:
                # 需要prefill
                prefill_seqs.append(seq)
            elif seq.cached_kv_len < len(seq.token_ids):
                # 需要decode
                decode_seqs.append(seq)
        
        results = {}
        
        # 处理prefill
        if prefill_seqs:
            print(f"Processing {len(prefill_seqs)} prefill sequences")
            batch = ForwardBatch(
                foward_mode=ForwardMode.PREFILL,
                num_seqs=len(prefill_seqs),
                seqs=prefill_seqs,
                max_bs=len(prefill_seqs)
            )
            
            try:
                new_token_ids = self.executor.execute("execute_model", batch)
                
                for seq, new_token_id in zip(prefill_seqs, new_token_ids):
                    seq.token_ids.append(new_token_id)
                    seq.cached_kv_len = len(seq.token_ids) - 1  # 除了最新的token，其他都已缓存
                    seq.status = SequenceStatus.RUNNING
                    results[seq.seq_id] = seq.token_ids.copy()
                    
            except Exception as e:
                print(f"Error in prefill: {e}")
                for seq in prefill_seqs:
                    seq.status = SequenceStatus.FINISHED
        
        # 处理decode
        if decode_seqs:
            print(f"Processing {len(decode_seqs)} decode sequences")
            batch = ForwardBatch(
                foward_mode=ForwardMode.DECODE,
                num_seqs=len(decode_seqs),
                seqs=decode_seqs,
                max_bs=len(decode_seqs)
            )
            
            try:
                new_token_ids = self.executor.execute("execute_model", batch)
                
                for seq, new_token_id in zip(decode_seqs, new_token_ids):
                    seq.token_ids.append(new_token_id)
                    seq.cached_kv_len = len(seq.token_ids) - 1
                    
                    # 检查是否达到最大长度或遇到结束token
                    if (len(seq.token_ids) >= len(seq.kv_indices) or 
                        len(seq.token_ids) - seq.num_tokens >= seq.sampling_params.max_new_tokens):
                        seq.status = SequenceStatus.FINISHED
                        print(f"Sequence {seq.seq_id} finished generation")
                    
                    results[seq.seq_id] = seq.token_ids.copy()
                    
            except Exception as e:
                print(f"Error in decode: {e}")
                for seq in decode_seqs:
                    seq.status = SequenceStatus.FINISHED
        
        return results
    
    def is_finished(self, seq_id: int) -> bool:
        """检查序列是否已完成"""
        seq = self.sequences.get(seq_id)
        return seq is None or seq.status == SequenceStatus.FINISHED
    
    def get_sequence(self, seq_id: int) -> Optional[Sequence]:
        """获取序列"""
        return self.sequences.get(seq_id)