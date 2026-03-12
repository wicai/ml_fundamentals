# Implement Sigmoid Activation

**Category:** coding
**Difficulty:** 2
**Tags:** coding, activations, numerical stability

## Question
Implement the sigmoid activation function from scratch with proper numerical stability.

Your implementation should include:
1. **`sigmoid`**: Numerically stable sigmoid that handles large positive AND large negative inputs
2. **`sigmoid_backward`**: The gradient of sigmoid (used in backprop)

Key insight: the naive `1 / (1 + exp(-x))` overflows for large negative x because `exp(-(-1000)) = exp(1000) = inf`.

**Function signature:**
```python
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable sigmoid: sigma(x) = 1 / (1 + exp(-x))

    Args:
        x: input tensor (any shape)
    Returns:
        output: same shape as x, values in (0, 1)
    """
    pass

def sigmoid_backward(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient of sigmoid: d(sigma)/dx = sigma(x) * (1 - sigma(x))

    Args:
        x: input tensor (any shape)
    Returns:
        grad: same shape as x
    """
    pass
```

## Answer

**Key concepts:**
1. Sigmoid: sigma(x) = 1 / (1 + exp(-x))
2. For large positive x: exp(-x) is tiny, so sigmoid ~ 1 (no problem)
3. For large negative x: exp(-x) overflows! Use identity: sigma(x) = exp(x) / (1 + exp(x))
4. Gradient: sigma'(x) = sigma(x) * (1 - sigma(x)), which is maximized at x=0

**Reference implementation:**
```python
import torch

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable sigmoid.

    Uses two formulas depending on sign of x:
    - x >= 0: 1 / (1 + exp(-x))      (exp(-x) is small, no overflow)
    - x < 0:  exp(x) / (1 + exp(x))  (exp(x) is small, no overflow)
    """
    pos_mask = x >= 0
    neg_mask = ~pos_mask

    result = torch.zeros_like(x)

    # For positive x: standard formula
    result[pos_mask] = 1.0 / (1.0 + torch.exp(-x[pos_mask]))

    # For negative x: equivalent formula that avoids overflow
    exp_x = torch.exp(x[neg_mask])
    result[neg_mask] = exp_x / (1.0 + exp_x)

    return result

# Alternative: using torch.clamp (simpler but less precise)
def sigmoid_clamp(x: torch.Tensor) -> torch.Tensor:
    """Simpler version using clamp to prevent overflow."""
    x_clamped = torch.clamp(x, min=-50, max=50)
    return 1.0 / (1.0 + torch.exp(-x_clamped))

# Alternative: using the where function (more elegant)
def sigmoid_where(x: torch.Tensor) -> torch.Tensor:
    """Using torch.where for branchless computation."""
    pos = 1.0 / (1.0 + torch.exp(-x.clamp(min=0)))
    neg_exp = torch.exp(x.clamp(max=0))
    neg = neg_exp / (1.0 + neg_exp)
    return torch.where(x >= 0, pos, neg)

def sigmoid_backward(x: torch.Tensor) -> torch.Tensor:
    """
    Gradient of sigmoid: sigma'(x) = sigma(x) * (1 - sigma(x))

    Note: the max gradient is 0.25, occurring at x=0.
    This is why deep sigmoid networks suffer from vanishing gradients -
    each layer multiplies gradients by at most 0.25.
    """
    s = sigmoid(x)
    return s * (1 - s)
```

**Testing:**
```python
import torch

# Test 1: Basic values
print("=" * 70)
print("TEST 1: Basic Sigmoid Values")
print("=" * 70)
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
result = sigmoid(x)
expected = torch.sigmoid(x)
print(f"Input:    {x.tolist()}")
print(f"Ours:     {result.tolist()}")
print(f"PyTorch:  {expected.tolist()}")
print(f"Match: {torch.allclose(result, expected)}")

# Test 2: Numerical stability with extreme values
print("\n" + "=" * 70)
print("TEST 2: Numerical Stability")
print("=" * 70)
extreme = torch.tensor([-1000.0, -100.0, 0.0, 100.0, 1000.0])
result = sigmoid(extreme)
expected = torch.sigmoid(extreme)
print(f"Input:    {extreme.tolist()}")
print(f"Ours:     {result.tolist()}")
print(f"PyTorch:  {expected.tolist()}")
print(f"No NaN: {not torch.isnan(result).any()}")
print(f"No Inf: {not torch.isinf(result).any()}")
print(f"All in (0,1): {(result >= 0).all() and (result <= 1).all()}")

# Test 3: Sigmoid properties
print("\n" + "=" * 70)
print("TEST 3: Properties")
print("=" * 70)
print(f"sigmoid(0) = {sigmoid(torch.tensor(0.0)):.4f} (should be 0.5)")
print(f"sigmoid(x) + sigmoid(-x) = {(sigmoid(torch.tensor(5.0)) + sigmoid(torch.tensor(-5.0))):.4f} (should be 1.0)")

# Test 4: Gradient
print("\n" + "=" * 70)
print("TEST 4: Gradient")
print("=" * 70)
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
grad = sigmoid_backward(x)
print(f"Input:    {x.tolist()}")
print(f"Gradient: {[f'{g:.4f}' for g in grad.tolist()]}")
print(f"Max grad at x=0: {grad[2]:.4f} (should be 0.25)")
print(f"Grad symmetric: {torch.allclose(grad, grad.flip(0))}")

# Test 5: Compare gradient with autograd
print("\n" + "=" * 70)
print("TEST 5: Gradient vs Autograd")
print("=" * 70)
x = torch.tensor([-1.0, 0.0, 1.0, 3.0], requires_grad=True)
y = torch.sigmoid(x)
y.sum().backward()
autograd_result = x.grad.clone()
our_result = sigmoid_backward(x.detach())
print(f"Autograd: {autograd_result.tolist()}")
print(f"Ours:     {our_result.tolist()}")
print(f"Match: {torch.allclose(autograd_result, our_result)}")
```

**Common mistakes:**
1. Not handling large negative inputs (exp overflow)
2. Forgetting that sigmoid(0) = 0.5 (not 0 or 1)
3. Wrong gradient formula (forgetting the `1 - sigma` term)
4. Not recognizing the vanishing gradient problem: max gradient is only 0.25

## Follow-up Questions
- Why do modern architectures avoid sigmoid in hidden layers?
- What's the relationship between sigmoid and softmax for 2 classes?
- How does the vanishing gradient problem relate to sigmoid's derivative?
- When IS sigmoid still used? (output layer for binary classification, gates in LSTM/GRU)
