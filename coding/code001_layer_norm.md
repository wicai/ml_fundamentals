# Implement Layer Normalization

**Category:** coding
**Difficulty:** 3
**Tags:** coding, normalization, architecture

## Question
Implement layer normalization in PyTorch from scratch (don't use `nn.LayerNorm`).

Your implementation should:
- Normalize across the last dimension (features)
- Support learnable scale (gamma) and shift (beta) parameters
- Handle numerical stability
- Match the behavior of `torch.nn.LayerNorm`

**Function signature:**
```python
from typing import Union
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, tuple], eps: float = 1e-5) -> None:
        """
        Args:
            normalized_shape: int or tuple, the shape to normalize over
            eps: float, epsilon for numerical stability
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (..., normalized_shape)
        Returns:
            normalized tensor of same shape as x
        """
        pass
```

## Answer

**Key concepts:**
1. Normalize across features (last dim) for each sample
2. Compute mean and variance per sample
3. Scale and shift with learnable parameters
4. Add epsilon for numerical stability

**Reference implementation:**
```python
from typing import Union
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, tuple], eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean and variance across last dimension
        # keepdim=True maintains shape for broadcasting
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        out = self.gamma * x_norm + self.beta

        return out
```

**Testing your implementation:**
```python
# Test
batch_size, seq_len, d_model = 2, 10, 512
x = torch.randn(batch_size, seq_len, d_model)

# Your implementation
ln_custom = LayerNorm(d_model)

# PyTorch's implementation
ln_pytorch = nn.LayerNorm(d_model)

# Copy parameters to match
ln_pytorch.weight.data = ln_custom.gamma.data
ln_pytorch.bias.data = ln_custom.beta.data

# Compare outputs
out_custom = ln_custom(x)
out_pytorch = ln_pytorch(x)

print(f"Max difference: {(out_custom - out_pytorch).abs().max().item():.2e}")
# Should be very small (< 1e-6)
```

**Common mistakes:**
1. ❌ Using `var(unbiased=True)` - LayerNorm uses biased variance (population variance)
2. ❌ Forgetting `eps` in sqrt - causes NaN when variance is 0
3. ❌ Wrong dimension for mean/var - should be over features (dim=-1), not batch
4. ❌ Not using `keepdim=True` - breaks broadcasting

**Why it works:**
```
Input: [batch, seq, features]

mean = [batch, seq, 1]  # mean over features
var = [batch, seq, 1]   # variance over features

x_norm = (x - mean) / sqrt(var + eps)  # Broadcasting works
# -> [batch, seq, features]

out = gamma * x_norm + beta  # gamma, beta: [features]
# -> [batch, seq, features]
```

## Follow-up Questions
- What's the difference between LayerNorm and BatchNorm?
- Why does LayerNorm use unbiased=False for variance?
- How would you implement RMSNorm (used in Llama)?
