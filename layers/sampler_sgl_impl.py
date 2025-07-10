import torch
from torch import nn

class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor | None = None,
        min_ps: torch.Tensor | None = None,
        top_ps: torch.Tensor | None = None,
        top_ks: torch.Tensor | None = None,
    ):
        """Run a sampler & compute logprobs and update logits_output accordingly.

        Args:
            logits_output: The logits from the model forward
        """

        is_all_greedy = (
            temperatures is None and
            min_ps is None and
            top_ps is None and
            top_ks is None
        )

        if is_all_greedy:
            # Use torch.argmax if all requests use greedy sampling
            batch_next_token_ids = torch.argmax(logits, -1)

        else:
            assert (
                temperatures is not None
                and min_ps is not None
                and top_ps is not None
                and top_ks is not None
            )
            # Post process logits
            logits.div_(temperatures)
            logits[:] = torch.softmax(logits, dim=-1)
            probs = logits
            del logits

            batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs(
                probs,
                top_ks,
                top_ps,
                min_ps,
            )

        return batch_next_token_ids


def top_k_top_p_min_p_sampling_from_probs(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
):
    """A top-k, top-p and min-p sampling implementation with native pytorch operations."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    min_p_thresholds = probs_sort[:, 0] * min_ps
    probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0

    sampled_index = torch.multinomial(probs_sort, num_samples=1)
    # int32 range is enough to represent the token ids
    probs_idx = probs_idx.to(torch.int32)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids