# Implement RoPE (Rotary Positional Embeddings)
# ====================================================================
#
# Implement Rotary Positional Embeddings (RoPE) as used in Llama, GPT-NeoX, and other modern LLMs.
# 
# RoPE applies a rotation to query and key vectors based on their position. This encodes relative position information directly into the attention mechanism.
# 
# **Function signature:**
#
# ====================================================================

import torch

def apply_rotary_pos_emb(x: torch.Tensor, position_ids: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """
    Apply rotary positional embeddings to input tensor.

    Args:
        x: input tensor of shape (batch, seq_len, num_heads, head_dim)
        position_ids: position indices, shape (batch, seq_len) or (seq_len,)
        theta: base for frequency computation
    Returns:
        tensor with RoPE applied, same shape as x
    """
    head_dim = x.shape[-1]
    # we wanna rotate x[0, 0, 0, 0] and x[0, 0, 0, 1], which is two scalars, by     
    # freqs = theta^(-2i/head_dim) = exp(log(theta^(-2i/head_dim)))
    # = exp(-2i/head_dim * log(theta))
    # dim: (head_dim//2)
    freqs = torch.exp(torch.arange(0, head_dim//2) * -2 * math.log(theta)/head_dim)
    # angles dim: (seq_len, head_dim//2)    
    angles = position_ids.unsqueeze(-1) @ freqs.view(1, head_dim//2)
    sin = torch.sin(angles).unsqueeze(-2) # dim (seq_len, 1, head_dim//2)
    cos = torch.cos(angles).unsqueeze(-2) # dim (seq_len, 1, head_dim//2)
    evens = x[:,:,:,0::2] # dim: (batch, seq_len, n_head, head_dim//2)
    odds = x[:,:,:,1::2]
    x_evens = (cos * evens - sin * odds) # dim: (batch, seq_len, n_head, head_dim//2)
    x_odds = (sin * evens + cos * odds)
    return torch.stack([x_evens, x_odds], dim=-1).flatten(-2)    

