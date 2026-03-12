# Extract Token-Level Log-Probabilities
# ====================================================================
#
# Given a language model and a token sequence, extract per-token log-probabilities and compute sequence-level scores. This is foundational for reward modeling, DPO, PPO surrogate objectives, perplexity, and safety evaluations.
# 
# Your implementation should include:
# 1. **`get_per_token_logprobs`**: Extract log P(token_t | tokens_<t) for each token
# 2. **`sequence_logprob`**: Sum per-token log-probs to get sequence-level log-probability
# 3. **`sequence_perplexity`**: Compute perplexity from per-token log-probs
# 4. **`compare_sequences`**: Given a prompt and two completions, which does the model prefer?
# 
# **Function signature:**
#
# ====================================================================

def get_per_token_logprobs(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Extract per-token log-probabilities from a causal language model.

    For each position t, compute log P(input_ids[t] | input_ids[0:t]).
    Position 0 has no conditioning, so we only return positions 1..T-1.

    Args:
        model: causal LM that returns logits of shape (batch, seq_len, vocab_size)
        input_ids: token IDs, shape (batch_size, seq_len)
    Returns:
        token_logprobs: shape (batch_size, seq_len - 1)
            token_logprobs[b, t] = log P(input_ids[b, t+1] | input_ids[b, 0:t+1])
    """
    import torch.nn.functional as F
    with torch.no_grad():
        token_logits = model(input_ids) # (batch, seq_len, vocab_size)
        token_logprobs = F.log_softmax(token_logits[:,:-1,:], dim=-1) # (batch, seq_len-1, vocab_size)
        # we need to use input_ids[:, 1:] to select the parts of token_logprobs
        # to use gather the axes need to be the same 
        return token_logprobs.gather(-1, input_ids[:,1:].unsqueeze(-1)).squeeze(-1)



def sequence_logprob(token_logprobs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute sequence-level log-probability by summing per-token log-probs.

    Args:
        token_logprobs: shape (batch_size, seq_len)
        mask: optional bool mask, shape (batch_size, seq_len). True = include token.
              Use this to mask out prompt tokens (only score the completion).
    Returns:
        seq_logprobs: shape (batch_size,)
    """
    if mask is not None:
        token_logprobs = token_logprobs.masked_fill(~mask, 0)
    return torch.sum(token_logprobs, dim=-1)    

def sequence_perplexity(token_logprobs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute per-sequence perplexity: exp(-1/N * sum(log_probs)).

    Args:
        token_logprobs: shape (batch_size, seq_len)
        mask: optional bool mask, shape (batch_size, seq_len)
    Returns:
        perplexity: shape (batch_size,)
    """
    if mask is not None:
        tokens_per_seq = torch.sum(mask.float(), dim=-1) # (batch_size, ) # N included per batch
    else:
        tokens_per_seq = token_logprobs.shape[-1]
    sequence_logprobs = sequence_logprob(token_logprobs, mask) # (batch_size,)
    perplexity = torch.exp(-1.0 * sequence_logprobs / tokens_per_seq)
    return perplexity


def compare_sequences(model: nn.Module, prompt_ids: torch.Tensor, completion_a_ids: torch.Tensor, completion_b_ids: torch.Tensor) -> dict[str, float]:
    """
    Given a prompt and two completions, compute which the model assigns higher probability.

    Args:
        model: causal LM
        prompt_ids: shape (1, prompt_len)
        completion_a_ids: shape (1, completion_a_len)
        completion_b_ids: shape (1, completion_b_len)
    Returns:
        dict with 'logprob_a', 'logprob_b', 'perplexity_a', 'perplexity_b', 'preferred' ('a' or 'b')
    """
    # concat prompt_ids with completion_a_ids
    input_ids_a = torch.cat([prompt_ids, completion_a_ids], dim=-1) # (1, prompt_len + completion_a_len)
    input_ids_b = torch.cat([prompt_ids, completion_b_ids], dim=-1) # (1, prompt_len + completion_b_len)
    # create the mask, 1 means include token
    prompt_len = prompt_ids.shape[-1]
    completion_a_len = completion_a_ids.shape[-1]
    completion_b_len = completion_b_ids.shape[-1]
    mask_a = torch.cat([torch.zeros(1, prompt_len-1, dtype=torch.bool), torch.ones(1, completion_a_len, dtype=torch.bool)], dim=-1)
    mask_b = torch.cat([torch.zeros(1, prompt_len-1, dtype=torch.bool), torch.ones(1, completion_b_len, dtype=torch.bool)], dim=-1)
    # get token logprobs from get_per_token_logprobs
    # dim (batch_size, seq_len - 1)
    token_logprobs_a = get_per_token_logprobs(model, input_ids_a)
    token_logprobs_b = get_per_token_logprobs(model, input_ids_b)
    # compute perplexity using token logprobs, mask 
    perplexity_a = sequence_perplexity(token_logprobs_a, mask_a)
    perplexity_b = sequence_perplexity(token_logprobs_b, mask_b)
    # perplexity of 1 is perfect, higher is worse
    preferred = 'b' if perplexity_a > perplexity_b else 'a'
    return {
        'logprob_a': sequence_logprob(token_logprobs_a, mask_a).item(),
        'logprob_b': sequence_logprob(token_logprobs_b, mask_b).item(),
        'perplexity_a': torch.mean(perplexity_a).item(),
        'perplexity_b': torch.mean(perplexity_b).item(),
        'preferred': preferred
    }

