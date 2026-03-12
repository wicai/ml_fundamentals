# Implement AdamW Optimizer Step
# ====================================================================
#
# Implement a single optimization step of the AdamW optimizer.
# 
# Your implementation should:
# - Update first moment (mean) and second moment (variance) estimates
# - Apply bias correction
# - Include weight decay (decoupled from gradient)
# - Match PyTorch's AdamW behavior
# 
# **Function signature:**
#
# ====================================================================

import torch

def adamw_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01
) -> None:
    """
    Perform one AdamW optimization step.

    Args:
        param: parameter tensor to update (modified in-place)
        grad: gradient tensor
        m: first moment estimate (modified in-place)
        v: second moment estimate (modified in-place)
        t: timestep (starts at 1)
        lr: learning rate
        beta1: exponential decay rate for first moment
        beta2: exponential decay rate for second moment
        eps: small constant for numerical stability
        weight_decay: weight decay coefficient
    Returns:
        None (modifies param, m, v in-place)
    """

    # We are doing adamW optimization which means we are moving param in a direction set by (grad, momentum, second moment)
    # normal gradient descent is just param + grad * lr 
    # here instead we are gonna keep track of momentum, which is a rolling average of the gradients
    m.multiply_(beta1).add_(grad * (1-beta1)) # update first moment estimate        
    v.multiply_(beta2).add_(grad ** 2 * (1-beta2))
    # when we actually use the first moment estimate to compute how far we're moving we are gonna need to do some telescoping sum of beta1    
    # toy example: at timestep 2, we need to divide it by (.1 * .9 + .1) 
    denom_m = 1
    for i in range(t):
        denom_m += beta1 ** i
    denom_m *= (1 - beta1)
    
    denom_v = 1
    for i in range(t):
        denom_v += beta2 ** i
    denom_v *= (1 - beta2)
    adj_m = m/denom_m
    adj_v = v/denom_v
    # not sure what the exact formula is to take a step but 
    # the intuition here is that is v is very large, we'd like to make our step a little smaller, since we're moving over a hill
    # if v is smaller, we can step a little faster since it's a long valley
    param.add_(adj_m * -lr / (torch.sqrt(adj_v) + eps))
    # weight decay
    param.multiply_(1 - lr * weight_decay)

