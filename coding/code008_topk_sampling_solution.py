# Implement Top-k and Top-p Sampling
# ====================================================================
#
# Implement top-k and top-p (nucleus) sampling for text generation.
# 
# Your implementation should:
# - Support top-k sampling (keep only top k highest probability tokens)
# - Support top-p sampling (keep tokens whose cumulative probability >= p)
# - Apply temperature scaling
# - Handle edge cases
# 
# **Function signature:**
#
# ====================================================================

from typing import Optional
import torch

def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> int:
    """
    Sample next token from logits with optional top-k/top-p filtering.

    Args:
        logits: shape (vocab_size,) - raw logits for next token
        temperature: float, controls randomness (higher = more random)
        top_k: int or None, keep only top k tokens
        top_p: float or None, keep tokens with cumulative prob >= p
    Returns:
        sampled token index (int)
    """    
    # temperature = 0.0 -> return max prob token, avoid div by 0 
    if temperature == 0:
        return torch.argmax(logits).item()
    
    logits = logits/temperature
    probs = torch.softmax(logits, dim=-1)        
    if top_k is not None:        
        top_k = min(top_k, logits.shape[0])
        # filter probs to top_k
        probs, indices_k = torch.topk(probs, top_k) # indices_k maps 0 to what token the 0th element of probs actually was 
        top_k_total_prob = torch.sum(probs)
        # renormalize so that probs add up to 1                
        probs.multiply_(1.0/top_k_total_prob)        
    if top_p is not None:        
        # go through until cumulative prob >= p
        probs, indices_p = torch.sort(probs, descending=True)         
        cumsum = torch.cumsum(probs, dim=0)
        cutoff = torch.searchsorted(cumsum, top_p).item()
        probs = probs[:cutoff+1]
        sum_probs = torch.sum(probs)
        probs.multiply_(1.0/sum_probs)
    sampled_pos = torch.multinomial(probs, 1)
    # if top_p, need to pass it through indices_p
    if top_p is not None:
        sampled_pos = indices_p[sampled_pos]
    if top_k is not None:
        sampled_pos = indices_k[sampled_pos]
    return sampled_pos.item()    

