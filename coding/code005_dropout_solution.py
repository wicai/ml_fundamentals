# Implement Dropout
# ====================================================================
#
# Implement dropout regularization from scratch in PyTorch.
# 
# Your implementation should:
# - Randomly zero out elements during training
# - Scale remaining elements to maintain expected value
# - Support training vs evaluation mode
# - Match PyTorch's `F.dropout` behavior
# 
# **Function signature:**
#
# ====================================================================

import torch

def dropout(x: torch.Tensor, p: float = 0.5, training: bool = True) -> torch.Tensor:
    """
    Apply dropout regularization.

    Args:
        x: input tensor of any shape
        p: probability of dropping an element (0 to 1)
        training: if True, apply dropout; if False, return input unchanged
    Returns:
        tensor of same shape as x with dropout applied
    """
    pass

