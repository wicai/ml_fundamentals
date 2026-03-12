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
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps       
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (..., normalized_shape)
        Returns:
            normalized tensor of same shape as x
        """
        n_dims = len(self.normalized_shape) if isinstance(self.normalized_shape, tuple) else 1
        dims_to_normalize = list(range(n_dims)) #0, 1, 2 
        dims_to_normalize = tuple([-1 * i - 1 for i in dims_to_normalize])
        x_mu = x.mean(dim=dims_to_normalize, keepdim=True)
        x_var = x.var(dim=dims_to_normalize, keepdim=True, unbiased=False)
        x_norm = (x-x_mu) / torch.sqrt(x_var + self.eps)
        return x_norm * self.gamma + self.beta

