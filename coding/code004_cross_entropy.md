# Implement Cross-Entropy Loss

**Category:** coding
**Difficulty:** 2
**Tags:** coding, loss-functions, numerical-stability

## Question
Implement cross-entropy loss from scratch with numerical stability.

Your implementation should:
- Compute cross-entropy from logits (not probabilities)
- Handle numerical stability using log-softmax
- Support batched inputs
- Match PyTorch's `F.cross_entropy` behavior

**Function signature:**
```python
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
```

## Answer

**Key concepts:**
1. Use log-softmax for numerical stability
2. Select the log-probability of the target class
3. Negate and average

**Reference implementation:**
```python
import torch
import torch.nn.functional as F

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Compute log-softmax (numerically stable)
    log_probs = F.log_softmax(logits, dim=-1)

    # Select log-probability of target class for each sample
    # Method 1: Using gather
    batch_size = logits.size(0)
    target_log_probs = log_probs[torch.arange(batch_size), targets]

    # Method 2: Using gather (alternative)
    # target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Negative log-likelihood, averaged over batch
    loss = -target_log_probs.mean()

    return loss

# Alternative: Manual log-softmax implementation
def cross_entropy_loss_manual(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Stable log-softmax
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted_logits = logits - max_logits
    exp_logits = torch.exp(shifted_logits)
    sum_exp = exp_logits.sum(dim=-1, keepdim=True)
    log_probs = shifted_logits - torch.log(sum_exp)

    # Select target log-probs
    batch_size = logits.size(0)
    target_log_probs = log_probs[torch.arange(batch_size), targets]

    # Negative log-likelihood
    loss = -target_log_probs.mean()

    return loss
```

**Testing:**
```python
# Test
batch_size, num_classes = 32, 10
logits = torch.randn(batch_size, num_classes)
targets = torch.randint(0, num_classes, (batch_size,))

# Your implementation
loss_custom = cross_entropy_loss(logits, targets)

# PyTorch's implementation
loss_pytorch = F.cross_entropy(logits, targets)

print(f"Custom loss: {loss_custom.item():.4f}")
print(f"PyTorch loss: {loss_pytorch.item():.4f}")
print(f"Difference: {abs(loss_custom - loss_pytorch).item():.2e}")

# Test numerical stability with large logits
large_logits = torch.tensor([[1000., 1001., 1002.]])
targets_large = torch.tensor([1])
loss_stable = cross_entropy_loss(large_logits, targets_large)
print(f"No overflow: {torch.isfinite(loss_stable)}")
```

**Common mistakes:**
1. ❌ Using softmax then log instead of log-softmax (numerically unstable)
2. ❌ Forgetting to negate the log-probabilities
3. ❌ Using sum instead of mean for batch reduction
4. ❌ Not handling numerical stability with large logits

## Follow-up Questions
- Why use log-softmax instead of softmax + log?
- What's the gradient of cross-entropy w.r.t. logits?
- How would you add label smoothing?
