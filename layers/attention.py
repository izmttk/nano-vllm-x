from contextlib import contextmanager
from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.kv_cache import KVCachePool
from core.common import ForwardMode, ForwardBatch

from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)

@dataclass
class AttentionMetadata:
    forward_mode: ForwardMode
    kv_cache: KVCachePool
    output_kv_indices: torch.Tensor
    # FlashInfer specific metadata
    prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper | None
    decode_wrapper: BatchDecodeWithPagedKVCacheWrapper | None

    @staticmethod
    def build(
        kv_cache: KVCachePool,
        batch: ForwardBatch,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device
    ):
        page_size = 1
        seqlens_q = torch.tensor(
            [len(seq.token_ids) - seq.cached_kv_len for seq in batch.seqs],
            dtype=torch.int32,
            device=device
        )  # (max_bs,)
        seqlens_kv = torch.tensor(
            [len(seq.kv_indices) for seq in batch.seqs],
            dtype=torch.int32,
            device=device
        )  # (max_bs,)

        workspace_size = 512 * 1024 * 1024
        workspace_buffer = torch.empty(
            workspace_size,
            dtype=torch.uint8,
            device=device,
        )

        qo_indptr = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(seqlens_q, dim=0, dtype=torch.int32)
        ]) # (max_bs + 1,)
        
        paged_seqlens_kv = seqlens_kv // page_size + (seqlens_kv % page_size != 0).int() # (max_bs,)
        paged_kv_indptr = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(paged_seqlens_kv, dim=0, dtype=torch.int32)
        ])  # (max_bs + 1,)
        paged_kv_last_page_len = seqlens_kv % page_size
        paged_kv_last_page_len = torch.where(paged_kv_last_page_len == 0,
                                             page_size, paged_kv_last_page_len) # (max_bs,)
    
        paged_kv_indices = []
        for seq, seq_paged_len in zip(batch.seqs, paged_seqlens_kv):
            seq_paged_len = int(seq_paged_len.item())
            seq_paged_kv = torch.tensor(
                seq.kv_indices + [-1] * (seq_paged_len - len(seq.kv_indices)),
                dtype=torch.int32,
                device=device
            )
            paged_kv_indices.append(seq_paged_kv)
        paged_kv_indices = torch.cat(paged_kv_indices, dim=0)  # (num_total_pages,)

        prefill_wrapper = None
        decode_wrapper = None
        if batch.forward_mode == ForwardMode.PREFILL:
            prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                workspace_buffer,
                "NHD",
                backend="auto",
            )
            prefill_wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indices=paged_kv_indices,
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_last_page_len=paged_kv_last_page_len,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                head_dim_vo=head_dim,
                page_size=page_size,
                causal=True,
                q_data_type=dtype
            )
        elif batch.forward_mode == ForwardMode.DECODE:
            decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer,
                "NHD",
                use_tensor_cores=False,
            )
            decode_wrapper.plan(
                indptr=paged_kv_indptr,
                indices=paged_kv_indices,
                last_page_len=paged_kv_last_page_len,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=page_size,
                q_data_type=dtype
            )
        
        output_kv_indices = torch.cat(
            [torch.tensor(
                seq.kv_indices[-(len(seq.kv_indices) - seq.cached_kv_len):],
                dtype=torch.long,
                device=device
            ) for seq in batch.seqs],
            dim=0
        )

        metadata = AttentionMetadata(
            forward_mode=batch.forward_mode,
            kv_cache=kv_cache,
            output_kv_indices=output_kv_indices,
            prefill_wrapper=prefill_wrapper,
            decode_wrapper=decode_wrapper,
        )
        return metadata


@contextmanager
def attention_kv_cache(model: nn.Module, metadata: AttentionMetadata):
    attn_modules: list[Attention] = []
    for module in model.modules():
        if isinstance(module, Attention):
            attn_modules.append(module)
            module.set_attention_metadata(metadata)
    yield
    for module in attn_modules:
        module.set_attention_metadata(None)

# flashinfer implemented attention
class Attention(nn.Module):
    def __init__(
        self,
        num_heads : int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads

        self.scaling = self.head_dim**-0.5 if scaling is None else scaling
        self.layer_id = layer_id

        self.attention_metadata: AttentionMetadata | None = None

    def set_attention_metadata(self, metadata: AttentionMetadata | None):
        self.attention_metadata = metadata

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        save_kv_cache=True
    ):
        assert self.attention_metadata is not None

        cache_loc = self.attention_metadata.output_kv_indices
        q = q.contiguous()

        if k is not None:
            assert v is not None
            if save_kv_cache:
                self.attention_metadata.kv_cache.set_kv_cache(
                    self.layer_id,
                    cache_loc,
                    k.view(-1, self.num_kv_heads, self.head_dim),
                    v.view(-1, self.num_kv_heads, self.head_dim)
                )

        # Call the wrapped function
        if self.attention_metadata.forward_mode == ForwardMode.PREFILL:
            assert self.attention_metadata.prefill_wrapper is not None
            o = self.attention_metadata.prefill_wrapper.forward(
                q.view(-1, self.num_heads, self.head_dim),
                self.attention_metadata.kv_cache.get_kv_cache(self.layer_id),
                causal=True
            )
        elif self.attention_metadata.forward_mode == ForwardMode.DECODE:
            assert self.attention_metadata.decode_wrapper is not None
            o = self.attention_metadata.decode_wrapper.forward(
                q.view(-1, self.num_heads, self.head_dim),
                self.attention_metadata.kv_cache.get_kv_cache(self.layer_id)
            )
        else:
            raise NotImplementedError


        return o.view(-1, self.num_heads * self.head_dim)
