# Implement Dropout

**Category:** coding
**Difficulty:** 2
**Tags:** coding, regularization, training

## Question
Implement dropout regularization from scratch in PyTorch.

Your implementation should:
- Randomly zero out elements during training
- Scale remaining elements to maintain expected value
- Support training vs evaluation mode
- Match PyTorch's `F.dropout` behavior

**Function signature:**
```python
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
```

## Answer

**Key concepts:**
1. Only apply during training
2. Generate random mask with probability p
3. Scale by 1/(1-p) to maintain expected value (inverted dropout)

**Reference implementation:**
```python
import torch

def dropout(x: torch.Tensor, p: float = 0.5, training: bool = True) -> torch.Tensor:
    # Don't apply dropout during evaluation
    if not training or p == 0:
        return x

    # Edge case: p=1 means drop everything
    if p == 1:
        return torch.zeros_like(x)

    # Generate random mask: 1 = keep, 0 = drop
    # Use Bernoulli distribution with probability (1-p) of keeping
    keep_prob = 1 - p
    mask = torch.bernoulli(torch.full_like(x, keep_prob))

    # Apply mask and scale by 1/(1-p) to maintain expected value
    # This is "inverted dropout" - scaling happens during training
    output = x * mask / keep_prob

    return output
```

**Alternative using binomial:**
```python
def dropout_alt(x: torch.Tensor, p: float = 0.5, training: bool = True) -> torch.Tensor:
    if not training or p == 0:
        return x
    if p == 1:
        return torch.zeros_like(x)

    keep_prob = 1 - p
    # Random mask: compare uniform random with keep_prob
    mask = (torch.rand_like(x) < keep_prob).float()

    return x * mask / keep_prob
```

**Testing:**
```python
# Test
x = torch.ones(1000, 100)
p = 0.5

# Training mode
out_train = dropout(x, p=p, training=True)

# Check dropout was applied (should be ~50% zeros)
dropout_rate = (out_train == 0).float().mean()
print(f"Dropout rate: {dropout_rate.item():.2f} (target: {p})")

# Check scaling maintains expected value
print(f"Mean before: {x.mean().item():.4f}")
print(f"Mean after: {out_train.mean().item():.4f}")  # Should be close

# Evaluation mode (should be unchanged)
out_eval = dropout(x, p=p, training=False)
print(f"Eval mode unchanged: {(out_eval == x).all()}")

# Compare with PyTorch
import torch.nn.functional as F
out_pytorch = F.dropout(x, p=p, training=True)
print(f"Both have same proportion of zeros: {((out_train == 0).sum() - (out_pytorch == 0).sum()).abs() < 100}")
```

**Common mistakes:**
1. ❌ Forgetting to scale by 1/(1-p) - changes expected value
2. ❌ Applying dropout during evaluation
3. ❌ Not using inverted dropout (scaling during training vs inference)
4. ❌ Wrong probability (using p instead of 1-p for Bernoulli)

## Follow-up Questions
- Why scale by 1/(1-p)?
- What's the difference between inverted dropout and standard dropout?
- How does dropout help prevent overfitting?
