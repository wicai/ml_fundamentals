# Implement Multi-Head Attention
# ====================================================================
#
# Implement multi-head attention from scratch in PyTorch.
# 
# Your implementation should:
# - Split into multiple attention heads
# - Compute parallel attention for each head
# - Concatenate and project outputs
# - Support optional masking
# 
# **Function signature:**
#
# ====================================================================

from typing import Optional
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        """
        Args:
            d_model: model dimension (must be divisible by num_heads)
            num_heads: number of attention heads
        """
        pass

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            Q, K, V: shape (batch, seq_len, d_model)
            mask: optional mask
        Returns:
            output: shape (batch, seq_len, d_model)
        """
        pass

