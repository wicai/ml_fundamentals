# Compute Perplexity for Language Models
# ====================================================================
#
# Implement perplexity computation for evaluating language models. Perplexity measures how "surprised" a model is by text — lower perplexity means the model predicts the text better.
# 
# Your implementation should include:
# 1. **`compute_perplexity`**: Compute perplexity of a model on a text sequence
# 2. **`compute_perplexity_batched`**: Compute over a dataset with proper handling of long sequences
# 3. **`compare_models`**: Compare two models on the same text
# 
# **Function signature:**
#
# exp(-1/n * sum_i logprob(token))
# ====================================================================

def compute_perplexity(model: nn.Module, input_ids: torch.Tensor) -> float:
    """
    Compute perplexity of a language model on input tokens.

    PPL = exp( -1/N * sum_{t=1}^{N} log P(token_t | token_{<t}) )

    Args:
        model: causal LM returning logits of shape (batch, seq_len, vocab_size)
        input_ids: token IDs, shape (1, seq_len)
    Returns:
        perplexity: scalar float (lower is better)
    """
    import torch.nn.functional as F
    logits = model(input_ids) # (1, seq_len, vocab_size)    
    logprobs = F.log_softmax(logits, dim=-1) # (1, seq_len, vocab_size)
    logprobs = logprobs[:,:-1,:].gather(2, input_ids[:,1:].unsqueeze(-1)).squeeze(-1) # dim (1, seq_len-1)
    n_logprobs = logprobs.shape[-1]
    return torch.exp(torch.sum(logprobs, dim=-1) * -1.0 / n_logprobs).item()

def compute_perplexity_batched(
    model: nn.Module,
    input_ids: torch.Tensor,
    stride: int = 512,
    max_length: int = 1024,
) -> float:
    """
    Compute perplexity on long text using a sliding window approach.

    For texts longer than max_length, use a sliding window with overlap.
    Only count each token once (using the stride region's predictions).

    Args:
        model: causal LM
        input_ids: shape (1, total_length) — can be very long
        stride: how far to slide the window each step
        max_length: context window size of the model
    Returns:
        perplexity: scalar float
    """
    total_length = input_ids.shape[-1]
    if total_length <= max_length:
        return compute_perplexity(model, input_ids)
    else:
        start_ind = 0
        end_ind = 0 
        running_logprob_sum = 0 
        running_n_tokens = 0 
        while end_ind < total_length:
            end_ind = min(start_ind+max_length, total_length) #end_ind 1024 | 1536
            inference_input = input_ids[:,start_ind:end_ind] #input_ids[1,0:1024] | [512:1536]
            logits = model(inference_input) # (1, 1024, vocab_size) | (1, 1024, vocab_size)
            logprobs = F.log_softmax(logits, dim=-1) #[1, 1024, vocab_size] | (1, 1024, vocab_size)
            # [1, 1023, vocab_size] -> [1, 1023] | [1, 1023, vocab_size] -> [1, 1023]
            selected_logprobs = logprobs[:,:-1,:].gather(2, input_ids[:, start_ind+1:end_ind].unsqueeze(-1)).squeeze(-1) #(1, seq_len-1)            
            if start_ind != 0:
                selected_logprobs = selected_logprobs[:,stride-1:] # [1, 1023-512]
                # in this second iteration is the selected_logprobs[0] here corresponding to token 1024 
                # (the 1025th token in input_ids), it is since start_ind+1+stride-1 = 512 + 512 + 1 = 1024
            running_n_tokens += selected_logprobs.shape[-1] # 1023 | 
            running_logprob_sum += torch.sum(selected_logprobs, dim=-1).item() 
            start_ind = start_ind + stride #512
        return math.exp(-1.0 * running_logprob_sum/running_n_tokens)

def compare_models(
    model_a: nn.Module,
    model_b: nn.Module,
    input_ids: torch.Tensor,
) -> dict[str, float]:
    """
    Compare perplexity of two models on the same text.

    Args:
        model_a, model_b: two causal LMs
        input_ids: shape (1, seq_len)
    Returns:
        dict with 'ppl_a', 'ppl_b', 'better' ('a' or 'b')
    """
    pass

