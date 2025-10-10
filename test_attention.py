from layers.attention import AttentionMetadata, Attention, attention_kv_cache
from core.common import ForwardBatch, ForwardMode, Sequence, SequenceStatus
from core.kv_cache import KVCachePool
import torch
import torch.nn.functional as F

num_tokens=8
num_layers=1
num_heads=1
head_dim=64
num_kv_heads=1
device=torch.device("cuda:0")

kv_cache = KVCachePool(
    dtype=torch.float16,
    device=device,
    num_tokens=num_tokens,
    num_layers=num_layers,
    num_heads=num_heads,
    head_dim=head_dim,
)

def get_prefill_batch():
    seq1 = Sequence(token_ids=[1,2,3], kv_indices=[0,1,2], cached_kv_len=2, status=SequenceStatus.RUNNING)
    seq2 = Sequence(token_ids=[4,5], kv_indices=[3,4], cached_kv_len=0, status=SequenceStatus.RUNNING)
    seq3 = Sequence(token_ids=[1,2,6,7,8], kv_indices=[0,1,5,6,7], cached_kv_len=2, status=SequenceStatus.RUNNING)
    
    
    for seq in [seq1, seq2, seq3]:
        for idx in seq.kv_indices[:seq.cached_kv_len]:
            kv_cache.set_kv_cache(
                0,
                torch.tensor([idx], dtype=torch.int32, device=device),
                torch.full(
                    (1, num_kv_heads, head_dim),
                    idx,
                    dtype=torch.float16,
                    device=device
                ),
                torch.full(
                    (1, num_kv_heads, head_dim),
                    1000+idx,
                    dtype=torch.float16,
                    device=device
                ),
            )
            
    batch = ForwardBatch(
        forward_mode=ForwardMode.PREFILL,
        num_seqs=3,
        seqs=[seq1, seq2, seq3],
        max_bs=3
    )
    return batch


batch = get_prefill_batch()
attention_metadata = AttentionMetadata.build(
    batch=batch,
    kv_cache=kv_cache,
    num_qo_heads=num_heads,
    num_kv_heads=num_kv_heads,
    head_dim=head_dim,
    dtype=torch.float16,
    device=device,
)

attn = Attention(
    layer_id=0,
    num_heads=num_heads,
    scaling=1.0 / (head_dim ** 0.5),
    head_dim=head_dim,
    num_kv_heads=num_kv_heads,
)
attn.set_attention_metadata(attention_metadata)


k_cache, v_cache = kv_cache.get_kv_cache(0)

print("==========================================================")
print("KV Cache before prefill:")
print(k_cache)
print(v_cache)

seqlen = sum([len(seq.token_ids) - seq.cached_kv_len for seq in batch.seqs])
q = torch.ones((seqlen, num_heads, head_dim), dtype=torch.float16, device=device)
k = torch.zeros((seqlen, num_kv_heads, head_dim), dtype=torch.float16, device=device)
v = torch.zeros((seqlen, num_kv_heads, head_dim), dtype=torch.float16, device=device)

for i, idx in enumerate(attention_metadata.output_kv_indices):
    idx = int(idx.item())
    k[i] = torch.full((num_kv_heads, head_dim), idx, dtype=torch.float16, device=device)
    v[i] = torch.full((num_kv_heads, head_dim), 1000+idx, dtype=torch.float16, device=device)

o = attn.forward(
    q=q,
    k=k,
    v=v,
)

print("==========================================================")
print("KV Cache after prefill:")
print(k_cache)
print(v_cache)

print("==========================================================")
print("Output:")
print(o)