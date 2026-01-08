# Implement GELU Activation

**Category:** coding
**Difficulty:** 2
**Tags:** coding, activations, transformers

## Question
Implement the GELU (Gaussian Error Linear Unit) activation function.

GELU is used in BERT, GPT, and many modern transformers. Implement both:
- Exact GELU using error function
- Tanh approximation (faster, commonly used)

**Function signature:**
```python
def gelu(x):
    """
    Exact GELU activation.

    Args:
        x: input tensor
    Returns:
        GELU(x)
    """
    pass

def gelu_approx(x):
    """
    GELU approximation using tanh.

    Args:
        x: input tensor
    Returns:
        approximate GELU(x)
    """
    pass
```

## Answer

**Key concepts:**
1. Exact GELU: x * Φ(x), where Φ is Gaussian CDF
2. Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
3. Tanh approximation for speed

**Reference implementation:**
```python
import torch
import math

def gelu(x):
    """
    Exact GELU: x * Φ(x)
    where Φ(x) is the Gaussian cumulative distribution function.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_approx(x):
    """
    GELU approximation using tanh.
    Used in the original BERT and GPT implementations.

    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    return 0.5 * x * (
        1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        )
    )

# Alternative: using sigmoid approximation
def gelu_sigmoid_approx(x):
    """
    Alternative GELU approximation using sigmoid.
    GELU(x) ≈ x * σ(1.702 * x)
    """
    return x * torch.sigmoid(1.702 * x)
```

**As a module:**
```python
class GELU(nn.Module):
    def __init__(self, approximate='none'):
        """
        Args:
            approximate: 'none' for exact, 'tanh' for tanh approximation
        """
        super().__init__()
        self.approximate = approximate

    def forward(self, x):
        if self.approximate == 'tanh':
            return gelu_approx(x)
        else:
            return gelu(x)
```

**Testing:**
```python
# Test
x = torch.linspace(-3, 3, 100)

# Exact GELU
y_exact = gelu(x)

# Tanh approximation
y_approx = gelu_approx(x)

# Sigmoid approximation
y_sigmoid = gelu_sigmoid_approx(x)

# PyTorch's GELU
y_pytorch = torch.nn.functional.gelu(x)
y_pytorch_approx = torch.nn.functional.gelu(x, approximate='tanh')

# Compare
print(f"Exact vs PyTorch exact: {(y_exact - y_pytorch).abs().max().item():.2e}")
print(f"Tanh approx vs PyTorch tanh: {(y_approx - y_pytorch_approx).abs().max().item():.2e}")
print(f"Exact vs Tanh approx max diff: {(y_exact - y_approx).abs().max().item():.4f}")

# Visualize properties
print(f"\nGELU(0) = {gelu(torch.tensor(0.)):.4f}")
print(f"GELU(-10) ≈ {gelu(torch.tensor(-10.)):.4f}")
print(f"GELU(10) ≈ {gelu(torch.tensor(10.)):.4f}")

# Benchmark
x_large = torch.randn(1000, 1000).cuda()

import time
# Exact GELU
start = time.time()
for _ in range(100):
    _ = gelu(x_large)
torch.cuda.synchronize()
time_exact = time.time() - start

# Approx GELU
start = time.time()
for _ in range(100):
    _ = gelu_approx(x_large)
torch.cuda.synchronize()
time_approx = time.time() - start

print(f"\nExact GELU: {time_exact:.4f}s")
print(f"Approx GELU: {time_approx:.4f}s")
print(f"Speedup: {time_exact / time_approx:.2f}x")
```

**Comparison with other activations:**
```python
x = torch.linspace(-3, 3, 100)

activations = {
    'ReLU': torch.relu(x),
    'GELU': gelu(x),
    'Swish/SiLU': x * torch.sigmoid(x),
    'Tanh': torch.tanh(x),
}

# GELU is smooth and non-monotonic for negative values
print("Comparison at x=-1:")
for name, y in activations.items():
    idx = (x == -1).nonzero()[0]
    print(f"{name:10s}: {y[idx].item():.4f}")
```

**Common mistakes:**
1. ❌ Wrong constant in approximation (0.044715)
2. ❌ Forgetting sqrt(2/π) in tanh version
3. ❌ Using ReLU formula by mistake
4. ❌ Not dividing by sqrt(2) in exact version

## Follow-up Questions
- Why use GELU instead of ReLU?
- What's the advantage of GELU's smoothness?
- Which modern models use GELU vs SwiGLU?
