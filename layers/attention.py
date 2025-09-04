from contextlib import contextmanager
from typing import Optional
from dataclasses import dataclass
import enum
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.kv_cache import KVCachePool
from core.common import ForwardMode, ForwardBatch

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

@dataclass
class AttentionMetadata:
    forward_mode: ForwardMode

    kv_cache: KVCachePool
    output_kv_indices: torch.Tensor
    seqlens_q: torch.Tensor # (max_bs,)
    seqlens_kv: torch.Tensor # (max_bs,)

    # Flash attention specific metadata for batch processing
    max_seqlen_q: int
    max_seqlen_kv: int
    cu_seqlens_q: torch.Tensor  # (max_bs + 1,)
    cu_seqlens_kv: torch.Tensor  # (max_bs + 1,)
    page_table: torch.Tensor  # (max_bs, max_num_blocks_per_seq)

    # Page-based KV cache metadata
    page_size: int
    
    @staticmethod
    def build(
        forward_mode: ForwardMode,
        kv_cache: KVCachePool,
        batch: ForwardBatch,
        device: torch.device
    ):
        page_size = 1
        seqlens_q = torch.tensor(
            [len(seq.token_ids) - seq.cached_kv_len for seq in batch.seqs],
            dtype=torch.long,
            device=device
        )
        seqlens_kv = torch.tensor(
            [seq.cached_kv_len for seq in batch.seqs],
            dtype=torch.long,
            device=device
        )
        
        max_seqlen_q = int(seqlens_q.max().item())
        max_seqlen_kv = int(seqlens_kv.max().item())
        
        # Create cumulative sequence length tensors for flash attention
        cu_seqlens_q = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            torch.cumsum(seqlens_q, dim=0)
        ])
        
        cu_seqlens_kv = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            torch.cumsum(seqlens_kv, dim=0)
        ])
        
        
        max_bs = len(seqlens_kv)
        max_num_blocks_per_seq = (max_seqlen_kv + page_size - 1) // page_size
        page_table = torch.zeros(
            (max_bs, max_num_blocks_per_seq),
            dtype=torch.long,
            device=device
        )
        for i, seq in enumerate(batch.seqs):
            kv_indices = torch.tensor(
                seq.kv_indices,
                dtype=torch.long,
                device=device
            )
            # Convert token indices to page indices
            page_indices = kv_indices // page_size
            # Get unique pages in order
            unique_pages = torch.unique_consecutive(page_indices)
            page_table[i, :len(unique_pages)] = unique_pages
            
        output_kv_indices_list: list[int] = []
        for seq in batch.seqs:
            output_kv_indices_list.extend(seq.kv_indices[seq.cached_kv_len:])
        output_kv_indices = torch.tensor(
            output_kv_indices_list,
            dtype=torch.long,
            device=device
        )
        
        metadata = AttentionMetadata(
            forward_mode=forward_mode,
            kv_cache=kv_cache,
            output_kv_indices=output_kv_indices,
            seqlens_q=seqlens_q,
            seqlens_kv=seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            page_table=page_table,
            page_size=page_size,
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


# Flash Attention implemented attention
class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: Optional[float],
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

    def _flash_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        causal: bool = True
    ) -> torch.Tensor:
        assert self.attention_metadata is not None
        
        if self.attention_metadata.forward_mode == ForwardMode.PREFILL:
            # q, k, v should be flattened to (total_tokens, num_heads, head_dim)
            q = q.contiguous().view(-1, self.num_heads, self.head_dim)
            # Get k, v from cache
            k_cache, v_cache = self.attention_metadata.kv_cache.get_kv_cache(self.layer_id)
            # Reshape to (num_pages, page_size, num_kv_heads, head_dim)
            k_cache = k_cache.view(
                -1, self.attention_metadata.page_size, self.num_kv_heads, self.head_dim
            )
            v_cache = v_cache.view(
                -1, self.attention_metadata.page_size, self.num_kv_heads, self.head_dim
            )
            
            output = flash_attn_varlen_func(
                q,
                k_cache,
                v_cache,
                block_table=self.attention_metadata.page_table,
                cu_seqlens_q=self.attention_metadata.cu_seqlens_q,
                cu_seqlens_k=self.attention_metadata.cu_seqlens_kv,
                max_seqlen_q=self.attention_metadata.max_seqlen_q,
                max_seqlen_k=self.attention_metadata.max_seqlen_kv,
                softmax_scale=self.scaling,
                causal=causal,
            )
            
            assert isinstance(output, torch.Tensor), "Flash attention should return a tensor"
            return output  # (total_tokens, num_heads, head_dim)
            
        elif self.attention_metadata.forward_mode == ForwardMode.DECODE:
            # (batch_size, 1, num_heads, head_dim)
            q = q.contiguous().view(-1, 1, self.num_heads, self.head_dim)
            # Get k, v cache in paged format
            k_cache, v_cache = self.attention_metadata.kv_cache.get_kv_cache(self.layer_id)
            # Get k, v from cache
            k_cache, v_cache = self.attention_metadata.kv_cache.get_kv_cache(self.layer_id)
            # Reshape to (num_pages, page_size, num_kv_heads, head_dim)
            k_cache = k_cache.view(
                -1, self.attention_metadata.page_size, self.num_kv_heads, self.head_dim
            )
            v_cache = v_cache.view(
                -1, self.attention_metadata.page_size, self.num_kv_heads, self.head_dim
            )
            
            output = flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                cache_seqlens=self.attention_metadata.seqlens_kv,
                block_table=self.attention_metadata.page_table,
                softmax_scale=self.scaling,
                causal=True,
            )
            
            assert isinstance(output, torch.Tensor), "Flash attention should return a tensor"
            return output.view(-1, self.num_heads, self.head_dim)
        else:
            raise ValueError(f"Unsupported forward mode: {self.attention_metadata.forward_mode}")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        save_kv_cache: bool = True
    ) -> torch.Tensor:
        assert self.attention_metadata is not None
        
        out_cache_loc = self.attention_metadata.output_kv_indices
        
        # Save K, V to cache if provided
        if k is not None and v is not None and save_kv_cache:
            self.attention_metadata.kv_cache.set_kv_cache(
                self.layer_id, out_cache_loc, k, v
            )
        
        output = self._flash_attention_forward(q, k, v, causal=True)
        return output
