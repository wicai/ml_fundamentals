# Implement Manual Backpropagation

**Category:** coding
**Difficulty:** 3
**Tags:** coding, backprop, gradients, training

## Question
Implement forward and backward passes manually for a simple 2-layer neural network, WITHOUT using `autograd`. This builds intuition for what `.backward()` actually does.

Network: `Input → Linear → ReLU → Linear → MSE Loss`

Your implementation should:
1. Implement the forward pass, caching intermediate values needed for backprop
2. Implement the backward pass, computing gradients for all weights and biases
3. Verify your gradients match PyTorch's autograd

**Function signature:**
```python
class ManualMLP:
    """
    2-layer MLP with manual forward and backward passes.
    Architecture: Linear(in, hidden) → ReLU → Linear(hidden, out)
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """Initialize weights (use torch tensors, but no autograd)."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Cache intermediate values for backward.

        Args:
            x: input, shape (batch_size, input_dim)
        Returns:
            output: shape (batch_size, output_dim)
        """
        pass

    def backward(self, grad_output: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Backward pass. Compute gradients for all parameters.

        Args:
            grad_output: gradient of loss w.r.t. output, shape (batch_size, output_dim)
        Returns:
            dict mapping parameter names to their gradients
        """
        pass

def mse_loss_and_grad(predictions: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute MSE loss and its gradient w.r.t. predictions.

    Args:
        predictions: shape (batch_size, output_dim)
        targets: shape (batch_size, output_dim)
    Returns:
        loss: scalar
        grad: shape (batch_size, output_dim), gradient of loss w.r.t. predictions
    """
    pass
```

## Answer

**Key concepts:**
1. Forward pass: compute output, cache values needed for backward
2. Backward pass: apply chain rule layer by layer in reverse order
3. For linear layer y = xW^T + b:
   - dL/dW = (dL/dy)^T @ x
   - dL/db = sum(dL/dy, dim=0)
   - dL/dx = dL/dy @ W  (to pass to previous layer)
4. For ReLU: gradient is 1 where input > 0, else 0
5. For MSE: dL/dy = 2 * (y - target) / N

**Reference implementation:**
```python
import torch

class ManualMLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        # Xavier initialization
        self.W1 = torch.randn(hidden_dim, input_dim) * (2 / (input_dim + hidden_dim)) ** 0.5
        self.b1 = torch.zeros(hidden_dim)
        self.W2 = torch.randn(output_dim, hidden_dim) * (2 / (hidden_dim + output_dim)) ** 0.5
        self.b2 = torch.zeros(output_dim)

        # Cache for backward pass
        self.cache = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1: z1 = x @ W1^T + b1
        z1 = x @ self.W1.T + self.b1
        self.cache['x'] = x
        self.cache['z1'] = z1

        # ReLU: a1 = max(0, z1)
        a1 = torch.clamp(z1, min=0)
        self.cache['a1'] = a1

        # Layer 2: z2 = a1 @ W2^T + b2
        z2 = a1 @ self.W2.T + self.b2
        self.cache['z2'] = z2

        return z2

    def backward(self, grad_output: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        grad_output: dL/dz2, shape (batch_size, output_dim)
        """
        x = self.cache['x']
        z1 = self.cache['z1']
        a1 = self.cache['a1']

        # Gradients for Layer 2 (z2 = a1 @ W2^T + b2)
        # dL/dW2 = grad_output^T @ a1
        dW2 = grad_output.T @ a1
        # dL/db2 = sum(grad_output, dim=0)
        db2 = grad_output.sum(dim=0)
        # dL/da1 = grad_output @ W2  (pass gradient to previous layer)
        da1 = grad_output @ self.W2

        # Gradient through ReLU: zero out where z1 <= 0
        dz1 = da1 * (z1 > 0).float()

        # Gradients for Layer 1 (z1 = x @ W1^T + b1)
        # dL/dW1 = dz1^T @ x
        dW1 = dz1.T @ x
        # dL/db1 = sum(dz1, dim=0)
        db1 = dz1.sum(dim=0)

        return {
            'W1': dW1,
            'b1': db1,
            'W2': dW2,
            'b2': db2,
        }

def mse_loss_and_grad(predictions: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    MSE = mean((pred - target)^2)
    dMSE/dpred = 2 * (pred - target) / N
    """
    diff = predictions - targets
    n = predictions.shape[0]

    loss = (diff ** 2).mean()
    grad = 2.0 * diff / n

    return loss, grad
```

**Testing:**
```python
import torch
import torch.nn as nn

torch.manual_seed(1)

# Test 1: Forward pass matches PyTorch
print("=" * 70)
print("TEST 1: Forward Pass Matches PyTorch")
print("=" * 70)

input_dim, hidden_dim, output_dim = 4, 8, 2
batch_size = 3

manual = ManualMLP(input_dim, hidden_dim, output_dim)

# Create equivalent PyTorch model
pytorch_model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim),
)
# Copy weights
with torch.no_grad():
    pytorch_model[0].weight.copy_(manual.W1)
    pytorch_model[0].bias.copy_(manual.b1)
    pytorch_model[2].weight.copy_(manual.W2)
    pytorch_model[2].bias.copy_(manual.b2)

x = torch.randn(batch_size, input_dim)
manual_out = manual.forward(x)
pytorch_out = pytorch_model(x)
print(f"Manual output:\n{manual_out}")
print(f"PyTorch output:\n{pytorch_out}")
print(f"Forward match: {torch.allclose(manual_out, pytorch_out, atol=1e-6)}")

# Test 2: Backward pass matches autograd
print("\n" + "=" * 70)
print("TEST 2: Gradients Match Autograd")
print("=" * 70)

x_auto = x.clone().detach()
targets = torch.randn(batch_size, output_dim)

# Manual backward
manual_out = manual.forward(x_auto)
loss, grad_output = mse_loss_and_grad(manual_out, targets)
grads = manual.backward(grad_output)

# PyTorch backward
x_pt = x.clone().detach()
pytorch_out = pytorch_model(x_pt)
loss_pt = nn.functional.mse_loss(pytorch_out, targets)
loss_pt.backward()

print(f"Loss — manual: {loss.item():.6f}, PyTorch: {loss_pt.item():.6f}")
print(f"Loss match: {torch.isclose(loss, loss_pt)}")
print()

for name, pt_param in [('W1', pytorch_model[0].weight),
                         ('b1', pytorch_model[0].bias),
                         ('W2', pytorch_model[2].weight),
                         ('b2', pytorch_model[2].bias)]:
    diff = (grads[name] - pt_param.grad).abs().max().item()
    match = torch.allclose(grads[name], pt_param.grad, atol=1e-5)
    print(f"d{name} max diff: {diff:.2e}, match: {match}")

# Test 3: Manual SGD step
print("\n" + "=" * 70)
print("TEST 3: Manual Training Step")
print("=" * 70)

lr = 0.01
manual.W1 -= lr * grads['W1']
manual.b1 -= lr * grads['b1']
manual.W2 -= lr * grads['W2']
manual.b2 -= lr * grads['b2']

# Check loss decreased
new_out = manual.forward(x_auto)
new_loss, _ = mse_loss_and_grad(new_out, targets)
print(f"Loss before step: {loss.item():.6f}")
print(f"Loss after step:  {new_loss.item():.6f}")
print(f"Loss decreased: {new_loss.item() < loss.item()}")
```

**Common mistakes:**
1. Wrong matrix dimensions in gradient computation (forgetting transpose)
2. Forgetting to cache values during forward pass
3. ReLU gradient: using `a1 > 0` instead of `z1 > 0` (same here, but conceptually z1 is correct)
4. MSE gradient: forgetting the `2/N` factor
5. Confusing `@` (matrix multiply) with `*` (element-wise)

## Follow-up Questions
- Why do we need to cache intermediate values during the forward pass?
- What happens to the gradient at exactly ReLU(0)? Does it matter in practice?
- How does autograd avoid explicit caching? (It builds a computation graph)
- How would you add a softmax + cross-entropy final layer?
- What is the gradient of sigmoid, and how does it cause vanishing gradients?
