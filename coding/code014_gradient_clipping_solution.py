# Implement Gradient Clipping
# ====================================================================
#
# Implement gradient clipping by global norm (as used in training LLMs).
# 
# Your implementation should:
# - Compute global norm across all parameters
# - Scale gradients if norm exceeds threshold
# - Support both clip by norm and clip by value
# 
# **Function signature:**
#
# ====================================================================

import torch
from typing import Iterable


def clip_grad_norm(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> float:
    """
    Clip gradients by global norm.

    Args:
        parameters: iterable of parameters (with .grad)
        max_norm: maximum gradient norm
    Returns:
        total_norm: the computed global gradient norm
    """

def clip_grad_value(parameters: Iterable[torch.nn.Parameter], clip_value: float) -> None:
    """
    Clip gradients by value.

    Args:
        parameters: iterable of parameters (with .grad)
        clip_value: maximum absolute value for gradients
    """             
    

