# Implement Softmax

**Category:** coding
**Difficulty:** 2
**Tags:** coding, basics, numerical-stability

## Question
Implement softmax in PyTorch from scratch with numerical stability.

Your implementation should:
- Handle numerical overflow (large positive values)
- Work with batched inputs
- Support specifying the dimension
- Match PyTorch's `F.softmax` behavior

**Function signature:**
```python
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
    pass
```

## Answer

**Key concepts:**
1. Subtract max for numerical stability (prevents overflow)
2. Exponentiate
3. Normalize by sum

**Reference implementation:**
```python
import torch

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Subtract max for numerical stability
    # This prevents overflow from exp(large_number)
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max

    # Compute exp
    exp_x = torch.exp(x_shifted)

    # Normalize
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp
```

**Testing:**
```python
# Test
x = torch.randn(2, 5, 10)

# Your implementation
out_custom = softmax(x, dim=-1)

# PyTorch's implementation
out_pytorch = torch.softmax(x, dim=-1)

print(f"Max difference: {(out_custom - out_pytorch).abs().max().item():.2e}")
print(f"Sum check (should be 1.0): {out_custom.sum(dim=-1)}")

# Test numerical stability
x_large = torch.tensor([[1000., 1001., 1002.]])
out_stable = softmax(x_large)
print(f"No overflow: {torch.isfinite(out_stable).all()}")
```

**Common mistakes:**
1. ❌ Not subtracting max - causes overflow with large numbers
2. ❌ Forgetting `keepdim=True` - breaks broadcasting
3. ❌ Wrong dimension for sum/max

## Follow-up Questions
- Why does subtracting the max improve numerical stability?
- What happens without the max subtraction when values are large?
- How is this different from log-softmax?
