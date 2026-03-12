# Implement Softmax
# ====================================================================
#
# Implement softmax in PyTorch from scratch with numerical stability.
# 
# Your implementation should:
# - Handle numerical overflow (large positive values)
# - Work with batched inputs
# - Support specifying the dimension
# - Match PyTorch's `F.softmax` behavior
# 
# **Function signature:**
#
# ====================================================================

import torch

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute softmax along specified dimension with numerical stability.

    Args:
        x: tensor of any shape
        dim: dimension to compute softmax over
    Returns:
        tensor of same shape as x with softmax applied
    """
    c = torch.max(x, dim=dim, keepdim=True).values
    x = x - c # numerically stable version of x we can exponentiate safely
    ex = torch.exp(x)
    sum_ex = torch.sum(ex, dim=dim, keepdim=True)
    return ex/sum_ex    

