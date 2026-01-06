"""
Implement Layer Normalization
======================================================================

Implement layer normalization in PyTorch from scratch (don't use `nn.LayerNorm`).

Your implementation should:
- Normalize across the last dimension (features)
- Support learnable scale (gamma) and shift (beta) parameters
- Handle numerical stability
- Match the behavior of `torch.nn.LayerNorm`

**Function signature:**
```python
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        """
        Args:
            normalized_shape: int or tuple, the shape to normalize over
            eps: float, epsilon for numerical stability
        """
        pass

    def forward(self, x):
        """
        Args:
            x: tensor of shape (..., normalized_shape)
        Returns:
            normalized tensor of same shape as x
        """
        pass
```
"""

# Your solution here:

