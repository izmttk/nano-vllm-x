import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: add kv cache
class Attention(nn.Module):
    def __init__(
        self,
        num_heads : int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads

        self.scaling = self.head_dim**-0.5 if scaling is None else scaling

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs,
    ):
        query = query.view(query.shape[:-1] + (self.num_heads, self.head_dim))
        key = key.view(key.shape[:-1] + (self.num_kv_heads, self.head_dim))
        value = value.view(value.shape[:-1] + (self.num_kv_heads, self.head_dim))

        query = query.transpose(-3, -2)
        key = key.transpose(-3, -2)
        value = value.transpose(-3, -2)

        attn_output = eager_attention_forward(
            self,
            query,
            key,
            value,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.transpose(-3, -2)
        attn_output = attn_output.reshape(attn_output.shape[:-2] + (self.num_heads * self.head_dim,))
        return attn_output
    

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    head_dim = hidden_states.shape[-1]
    slen = hidden_states.shape[-2]
    num_key_value_heads = hidden_states.shape[-3]
    other_dims = hidden_states.shape[:-3]

    if n_rep == 1:
        return hidden_states
    
    hidden_states = hidden_states.unsqueeze(-3)
    hidden_states = hidden_states.expand(*other_dims, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(*other_dims, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    is_causal: bool = True,
    attn_mask: torch.Tensor | None = None,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_kv_groups)
    value_states = repeat_kv(value, module.num_kv_groups)

    seq_len = query.size(-2)
    head_dim = query.size(-1)
    seq_len_kv = key_states.size(-2)

    attn_bias = torch.zeros(seq_len, seq_len_kv, dtype=query.dtype, device=query.device)

    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(seq_len, seq_len_kv, dtype=torch.bool).tril(diagonal=0).to(query.device)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    attn_weights = torch.matmul(query, key_states.transpose(-2, -1)) * scaling
    attn_weights += attn_bias
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    return attn_output