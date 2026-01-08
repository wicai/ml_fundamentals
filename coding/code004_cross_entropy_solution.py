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
    # We want to compute something of 
    #(batch_size,) which has for each item, 
    # sum_i^n 1_targets==i * -log(logits_i)

    # One way would be to select the ith score of logits, and reduce it down to batch_size
    # Another way would be to targets to be one hot instead of the class index, and then do some multiplication with logits to get the correct score, then do negative log
    
    pass

