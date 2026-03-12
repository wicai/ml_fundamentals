# Implement Sigmoid Activation
# ====================================================================
#
# Implement the sigmoid activation function from scratch with proper numerical stability.
# 
# Your implementation should include:
# 1. **`sigmoid`**: Numerically stable sigmoid that handles large positive AND large negative inputs
# 2. **`sigmoid_backward`**: The gradient of sigmoid (used in backprop)
# 
# Key insight: the naive `1 / (1 + exp(-x))` overflows for large negative x because `exp(-(-1000)) = exp(1000) = inf`.
# 
# **Function signature:**
#
# ====================================================================

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable sigmoid: sigma(x) = 1 / (1 + exp(-x))

    Args:
        x: input tensor (any shape)
    Returns:
        output: same shape as x, values in (0, 1)
    """
    # we're in trouble if x is large magnitude and neg
    return torch.where(x < 0, torch.exp(x)/(1+torch.exp(x)), 1/(1 + torch.exp(-x)) )
    
def sigmoid_backward(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient of sigmoid: d(sigma)/dx = sigma(x) * (1 - sigma(x))

    Args:
        x: input tensor (any shape)
    Returns:
        grad: same shape as x
    """
    pass

