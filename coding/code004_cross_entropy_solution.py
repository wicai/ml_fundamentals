# Implement Cross-Entropy Loss
# ====================================================================
#
# Implement cross-entropy loss from scratch with numerical stability.
# 
# Your implementation should:
# - Compute cross-entropy from logits (not probabilities)
# - Handle numerical stability using log-softmax
# - Support batched inputs
# - Match PyTorch's `F.cross_entropy` behavior
# 
# **Function signature:**
#
# ====================================================================

import torch

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss from logits.

    Args:
        logits: shape (batch_size, num_classes) - raw scores
        targets: shape (batch_size,) - class indices
    Returns:
        scalar loss value (mean over batch)
    """
    pass

