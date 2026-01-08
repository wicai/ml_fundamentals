# Implement Layer Normalization
# ====================================================================
#
# Implement layer normalization in PyTorch from scratch (don't use `nn.LayerNorm`).
# 
# Your implementation should:
# - Normalize across the last dimension (features)
# - Support learnable scale (gamma) and shift (beta) parameters
# - Handle numerical stability
# - Match the behavior of `torch.nn.LayerNorm`
# 
# **Function signature:**
#
# ====================================================================

from typing import Union
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, tuple], eps: float = 1e-5) -> None:
        """
        Args:
            normalized_shape: int or tuple, the shape to normalize over
            eps: float, epsilon for numerical stability
        """
        super().__init__()  # ✅ MUST call this first!
        self.normalized_shape = normalized_shape
        self.eps = eps        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (..., normalized_shape)
        Returns:
            normalized tensor of same shape as x
        """
        # Compute the mean and variance
        # To compute means, we need to compute the mean for each sample.
        # So if normalized_shape = (2,3) and x is (..., normalized_shape)
        norm_shape_dims = list(range(-len(self.normalized_shape), 0))
        x_means = x.mean(norm_shape_dims, keepdim=True) #(..., (1,1,...))
        x_vars = x.var(norm_shape_dims, keepdim=True, unbiased=False)  # ✅ Use biased variance

        # Normalize
        x_norm = (x - x_means) / torch.sqrt(x_vars + self.eps)

        # ✅ Apply scale (gamma) and shift (beta)
        return self.gamma * x_norm + self.beta
