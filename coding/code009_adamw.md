# Implement AdamW Optimizer Step

**Category:** coding
**Difficulty:** 3
**Tags:** coding, optimization, training

## Question
Implement a single optimization step of the AdamW optimizer.

Your implementation should:
- Update first moment (mean) and second moment (variance) estimates
- Apply bias correction
- Include weight decay (decoupled from gradient)
- Match PyTorch's AdamW behavior

**Function signature:**
```python
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
    pass
```

## Answer

**Key concepts:**
1. Update biased first moment: m = beta1 * m + (1 - beta1) * grad
2. Update biased second moment: v = beta2 * v + (1 - beta2) * grad^2
3. Bias correction: m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
4. Weight decay (decoupled): param = param * (1 - lr * weight_decay)
5. Update: param = param - lr * m_hat / (sqrt(v_hat) + eps)

**Reference implementation:**
```python
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
    # Update biased first moment estimate
    m.mul_(beta1).add_(grad, alpha=1 - beta1)

    # Update biased second moment estimate
    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    # Compute bias-corrected first moment
    bias_correction1 = 1 - beta1 ** t
    bias_correction2 = 1 - beta2 ** t

    m_hat = m / bias_correction1
    v_hat = v / bias_correction2

    # AdamW: Apply weight decay BEFORE gradient update (decoupled)
    if weight_decay != 0:
        param.mul_(1 - lr * weight_decay)

    # Update parameters
    param.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

# Alternative: More explicit version
def adamw_step_v2(
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
    # Update moments
    m[:] = beta1 * m + (1 - beta1) * grad
    v[:] = beta2 * v + (1 - beta2) * grad ** 2

    # Bias correction
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # Weight decay (decoupled from gradient)
    if weight_decay != 0:
        param[:] = param - lr * weight_decay * param

    # Gradient update
    param[:] = param - lr * m_hat / (torch.sqrt(v_hat) + eps)
```

**Full optimizer class:**
```python
from typing import Tuple, List, Iterator
import torch.nn as nn

class AdamW:
    def __init__(
        self,
        params: Iterator[nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize state
        self.state = []
        for param in self.params:
            self.state.append({
                'm': torch.zeros_like(param),
                'v': torch.zeros_like(param),
                't': 0
            })

    def step(self) -> None:
        for param, state in zip(self.params, self.state):
            if param.grad is None:
                continue

            state['t'] += 1

            adamw_step(
                param.data,
                param.grad.data,
                state['m'],
                state['v'],
                state['t'],
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                self.weight_decay
            )

    def zero_grad(self) -> None:
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
```

**Testing:**
```python
# Test
param = torch.randn(10, 5, requires_grad=True)
target = torch.randn(10, 5)

# Initialize optimizer state
m = torch.zeros_like(param)
v = torch.zeros_like(param)

# Simulate a few training steps
for t in range(1, 6):
    # Forward pass
    loss = ((param - target) ** 2).mean()

    # Backward pass
    loss.backward()

    # Optimizer step
    with torch.no_grad():
        adamw_step(param, param.grad, m, v, t, lr=0.01)

    # Zero gradients
    param.grad.zero_()

    print(f"Step {t}, Loss: {loss.item():.4f}")

# Compare with PyTorch's AdamW
param_pytorch = torch.randn(10, 5, requires_grad=True)
param_pytorch.data = param.data.clone()

optimizer = torch.optim.AdamW([param_pytorch], lr=0.01, betas=(0.9, 0.999), weight_decay=0.01)

loss = ((param_pytorch - target) ** 2).mean()
loss.backward()
optimizer.step()

print(f"\nParameter difference after 1 step: {(param - param_pytorch).abs().max().item():.2e}")
```

**Common mistakes:**
1. ❌ Applying weight decay to gradients (like L2 regularization) instead of parameters directly
2. ❌ Forgetting bias correction
3. ❌ Wrong order of operations for in-place updates
4. ❌ Not starting timestep t at 1

## Follow-up Questions
- What's the difference between AdamW and Adam with L2 regularization?
- Why is bias correction needed?
- When might you use a different beta2 value?
