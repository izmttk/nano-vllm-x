from contextlib import contextmanager
from typing import Optional
from dataclasses import dataclass
import enum
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.kv_cache import KVCachePool
from core.common import ForwardMode

from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)

@dataclass
class AttentionMetadata:
    forward_mode: ForwardMode

    kv_cache: KVCachePool
    kv_indices: torch.Tensor
    output_kv_indices: torch.Tensor

    workspace_buffer: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    qo_indptr: torch.Tensor
    prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper
    decode_wrapper: BatchDecodeWithPagedKVCacheWrapper
    

    @staticmethod
    def build(
        forward_mode: ForwardMode,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        kv_cache: KVCachePool,
        kv_indices: torch.Tensor,
        output_kv_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_seq_lens: torch.Tensor,
        device: torch.device
    ):
        flashinfer_workspace_size = 512 * 1024 * 1024

        # Allocate buffers
        workspace_buffer = torch.empty(
            flashinfer_workspace_size,
            dtype=torch.uint8,
            device=device,
        )
        qo_indptr = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            torch.cumsum(seq_lens, dim=0)
        ])
        paged_kv_indptr = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            torch.cumsum(kv_seq_lens // page_size * page_size, dim=0)
        ])
        paged_kv_indices = kv_indices[::page_size]
        paged_kv_last_page_len = kv_seq_lens % page_size
        paged_kv_last_page_len = torch.where(paged_kv_last_page_len == 0,
                                             page_size, paged_kv_last_page_len)

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
        )
        
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
        )
        metadata = AttentionMetadata(
            forward_mode=forward_mode,
            kv_cache=kv_cache,
            kv_indices=kv_indices,
            output_kv_indices=output_kv_indices,
            workspace_buffer=workspace_buffer,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            qo_indptr=qo_indptr,
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
                    self.layer_id, cache_loc, k, v
                )

        # Call the wrapped function
        if self.attention_metadata.forward_mode == ForwardMode.PREFILL:
            o = self.attention_metadata.prefill_wrapper.forward(
                q.view(-1, self.num_heads, self.head_dim),
                self.attention_metadata.kv_cache.get_kv_cache(self.layer_id),
                causal=True
            )
        elif self.attention_metadata.forward_mode == ForwardMode.DECODE:
            o = self.attention_metadata.decode_wrapper.forward(
                q.view(-1, self.num_heads, self.head_dim),
                self.attention_metadata.kv_cache.get_kv_cache(self.layer_id)
            )
        else:
            raise NotImplementedError


        return o.view(-1, self.num_heads * self.head_dim)
