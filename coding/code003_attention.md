# Implement Scaled Dot-Product Attention

**Category:** coding
**Difficulty:** 3
**Tags:** coding, attention, transformers

## Question
Implement scaled dot-product attention from scratch in PyTorch.

Your implementation should:
- Compute attention scores using Q, K, V matrices
- Apply scaling by sqrt(d_k)
- Support optional causal masking
- Handle batched inputs

**Function signature:**
```python
from typing import Optional, Tuple
import torch

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    Args:
        Q: queries, shape (batch, seq_len, d_k)
        K: keys, shape (batch, seq_len, d_k)
        V: values, shape (batch, seq_len, d_v)
        mask: optional mask, shape (seq_len, seq_len) or broadcastable
              True/1 means keep, False/0 means mask out
    Returns:
        output: shape (batch, seq_len, d_v)
        attention_weights: shape (batch, seq_len, seq_len)
    """
    pass
```

## Answer

**Key concepts:**
1. Compute attention scores: Q @ K^T
2. Scale by sqrt(d_k) for stable gradients
3. Apply mask before softmax (set to -inf)
4. Apply softmax and weighted sum with V

**Reference implementation:**
```python
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Get dimension for scaling
    d_k = Q.size(-1)

    # Compute attention scores: Q @ K^T
    # Shape: (batch, seq_len, d_k) @ (batch, d_k, seq_len)
    #     -> (batch, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Scale by sqrt(d_k)
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=scores.dtype))

    # Apply mask if provided (set masked positions to -inf)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)

    # Handle case where entire row is masked (results in NaN)
    attention_weights = attention_weights.masked_fill(
        torch.isnan(attention_weights), 0.0
    )

    # Weighted sum of values
    output = torch.matmul(attention_weights, V)

    return output, attention_weights
```

**Testing:**
```python
# Test
batch_size, seq_len, d_k, d_v = 2, 10, 64, 64
Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_v)

# No mask
output, attn = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")  # (2, 10, 64)
print(f"Attention weights sum: {attn.sum(dim=-1)}")  # Should be all 1s

# Causal mask
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
output_causal, attn_causal = scaled_dot_product_attention(Q, K, V, causal_mask)
print(f"Causal attention is lower triangular: {(attn_causal.tril() == attn_causal).all()}")
```

**Common mistakes:**
1. ❌ Forgetting to scale by sqrt(d_k)
2. ❌ Wrong transpose dimensions (should be -2, -1)
3. ❌ Masking with 0 instead of -inf
4. ❌ Not handling NaN from fully masked rows

## Follow-up Questions
- Why scale by sqrt(d_k)?
- What's the computational complexity?
- How does this differ from multi-head attention?
