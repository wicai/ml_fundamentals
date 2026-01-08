# Implement Gradient Clipping

**Category:** coding
**Difficulty:** 2
**Tags:** coding, training, optimization

## Question
Implement gradient clipping by global norm (as used in training LLMs).

Your implementation should:
- Compute global norm across all parameters
- Scale gradients if norm exceeds threshold
- Support both clip by norm and clip by value

**Function signature:**
```python
def clip_grad_norm(parameters, max_norm):
    """
    Clip gradients by global norm.

    Args:
        parameters: iterable of parameters (with .grad)
        max_norm: maximum gradient norm
    Returns:
        total_norm: the computed global gradient norm
    """
    pass

def clip_grad_value(parameters, clip_value):
    """
    Clip gradients by value.

    Args:
        parameters: iterable of parameters (with .grad)
        clip_value: maximum absolute value for gradients
    """
    pass
```

## Answer

**Key concepts:**
1. Global norm: sqrt(sum of squared norms across all params)
2. If total_norm > max_norm, scale all grads by max_norm / total_norm
3. Value clipping: clamp each gradient element independently

**Reference implementation:**
```python
import torch

def clip_grad_norm(parameters, max_norm):
    """
    Clip gradients by global norm (like torch.nn.utils.clip_grad_norm_).
    """
    parameters = list(parameters)

    # Filter out parameters without gradients
    parameters = [p for p in parameters if p.grad is not None]

    if len(parameters) == 0:
        return torch.tensor(0.)

    # Compute global norm
    # total_norm = sqrt(sum(||grad_i||^2 for all params i))
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([
            torch.norm(p.grad.detach(), 2) for p in parameters
        ]), 2
    )

    # Compute clipping coefficient
    clip_coef = max_norm / (total_norm + 1e-6)

    # If total_norm > max_norm, scale all gradients
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)

    return total_norm

# Alternative: more memory efficient
def clip_grad_norm_efficient(parameters, max_norm):
    """
    More memory-efficient version that doesn't create intermediate tensors.
    """
    parameters = [p for p in parameters if p.grad is not None]

    if len(parameters) == 0:
        return torch.tensor(0.)

    # Compute total norm without creating intermediate tensor
    total_norm = 0.
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    # Clip
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)

    return total_norm

def clip_grad_value(parameters, clip_value):
    """
    Clip gradients by value (clamp each element).
    """
    parameters = [p for p in parameters if p.grad is not None]

    for p in parameters:
        p.grad.detach().clamp_(-clip_value, clip_value)
```

**Testing:**
```python
# Test gradient clipping by norm
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

# Create some gradients
x = torch.randn(32, 10)
y = torch.randn(32, 1)
loss = ((model(x) - y) ** 2).mean()
loss.backward()

# Check gradient norm before clipping
total_norm_before = clip_grad_norm(model.parameters(), max_norm=float('inf'))
print(f"Gradient norm before clipping: {total_norm_before:.4f}")

# Clip gradients
max_norm = 1.0
total_norm_after = clip_grad_norm(model.parameters(), max_norm=max_norm)
print(f"Gradient norm after clipping: {total_norm_after:.4f}")

# Verify it's clipped
actual_norm = torch.norm(
    torch.stack([torch.norm(p.grad, 2) for p in model.parameters()]), 2
)
print(f"Actual norm after clipping: {actual_norm:.4f}")
print(f"Is clipped: {actual_norm <= max_norm}")

# Compare with PyTorch
model2 = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)
model2.load_state_dict(model.state_dict())
loss2 = ((model2(x) - y) ** 2).mean()
loss2.backward()

norm_pytorch = torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm)
print(f"\nPyTorch clip result: {norm_pytorch:.4f}")
print(f"Custom clip result: {total_norm_before:.4f}")
print(f"Difference: {abs(norm_pytorch - total_norm_before):.2e}")

# Test value clipping
model.zero_grad()
loss.backward()

clip_grad_value(model.parameters(), clip_value=0.1)
max_grad = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
print(f"\nMax gradient after value clipping: {max_grad:.4f}")
print(f"Is within bounds: {max_grad <= 0.1}")
```

**Usage in training loop:**
```python
def train_step(model, optimizer, x, y, max_grad_norm=1.0):
    # Forward pass
    output = model(x)
    loss = criterion(output, y)

    # Backward pass
    loss.backward()

    # Gradient clipping (before optimizer step!)
    total_norm = clip_grad_norm(model.parameters(), max_grad_norm)

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    return loss.item(), total_norm
```

**Common mistakes:**
1. ❌ Clipping after optimizer.step() instead of before
2. ❌ Computing norm incorrectly (not using global norm)
3. ❌ Forgetting to check if grad is not None
4. ❌ Creating unnecessary intermediate tensors (memory inefficient)

## Follow-up Questions
- When should you use gradient clipping?
- Why clip by norm instead of value?
- How do you choose the max_norm threshold?
