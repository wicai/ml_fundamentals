# Implement Scaled Dot-Product Attention
# ====================================================================
#
# Implement scaled dot-product attention from scratch in PyTorch.
# 
# Your implementation should:
# - Compute attention scores using Q, K, V matrices
# - Apply scaling by sqrt(d_k)
# - Support optional causal masking
# - Handle batched inputs
# 
# **Function signature:**
#
# ====================================================================

from typing import Optional, Tuple
import torch

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    Args:
        Q: queries, shape (batch, seq_len, d_k)
        K: keys, shape (batch, seq_len, d_k)
        V: values, shape (batch, seq_len, d_v)
        mask: optional mask, shape (seq_len, seq_len) or broadcastable
              True/1 means keep, False/0 means mask out
    Returns:
        output: shape (batch, seq_len, d_v)
        attention_weights: shape (batch, seq_len, seq_len)
    """
    pass

