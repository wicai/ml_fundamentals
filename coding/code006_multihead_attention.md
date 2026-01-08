# Implement Multi-Head Attention

**Category:** coding
**Difficulty:** 4
**Tags:** coding, attention, transformers, architecture

## Question
Implement multi-head attention from scratch in PyTorch.

Your implementation should:
- Split into multiple attention heads
- Compute parallel attention for each head
- Concatenate and project outputs
- Support optional masking

**Function signature:**
```python
from typing import Optional
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        """
        Args:
            d_model: model dimension (must be divisible by num_heads)
            num_heads: number of attention heads
        """
        pass

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            Q, K, V: shape (batch, seq_len, d_model)
            mask: optional mask
        Returns:
            output: shape (batch, seq_len, d_model)
        """
        pass
```

## Answer

**Key concepts:**
1. Linear projections for Q, K, V
2. Split into multiple heads (reshape)
3. Parallel scaled dot-product attention per head
4. Concatenate heads and final projection

**Reference implementation:**
```python
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        # Linear projections (can be done with single matrix per Q/K/V)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, d_k).
        Args:
            x: shape (batch, seq_len, d_model)
        Returns:
            shape (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        # Reshape to (batch, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose to (batch, num_heads, seq_len, d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of split_heads.
        Args:
            x: shape (batch, num_heads, seq_len, d_k)
        Returns:
            shape (batch, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        # Transpose to (batch, seq_len, num_heads, d_k)
        x = x.transpose(1, 2)
        # Reshape to (batch, seq_len, d_model)
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Linear projections
        Q = self.W_q(Q)  # (batch, seq_len, d_model)
        K = self.W_k(K)
        V = self.W_v(V)

        # Split into heads
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Scaled dot-product attention for all heads in parallel
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, num_heads, seq_len, seq_len)
        scores = scores / torch.sqrt(torch.tensor(self.d_k, dtype=scores.dtype))

        if mask is not None:
            # Expand mask for heads dimension
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = attention_weights.masked_fill(torch.isnan(attention_weights), 0.0)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # (batch, num_heads, seq_len, d_k)

        # Combine heads
        output = self.combine_heads(output)  # (batch, seq_len, d_model)

        # Final linear projection
        output = self.W_o(output)

        return output
```

**Testing:**
```python
# Test
batch_size, seq_len, d_model, num_heads = 2, 10, 512, 8

Q = torch.randn(batch_size, seq_len, d_model)
K = torch.randn(batch_size, seq_len, d_model)
V = torch.randn(batch_size, seq_len, d_model)

# Your implementation
mha = MultiHeadAttention(d_model, num_heads)
output = mha(Q, K, V)

print(f"Output shape: {output.shape}")  # Should be (2, 10, 512)
print(f"Parameters: {sum(p.numel() for p in mha.parameters())}")

# Test with causal mask
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
output_masked = mha(Q, K, V, causal_mask)
print(f"Masked output shape: {output_masked.shape}")
```

**Common mistakes:**
1. ❌ Wrong reshape/transpose for splitting heads
2. ❌ Not using contiguous() before final view
3. ❌ Wrong dimension for d_k (should be d_model / num_heads)
4. ❌ Forgetting final output projection W_o

## Follow-up Questions
- Why split into multiple heads instead of one large head?
- What's the computational complexity?
- How does this compare to grouped-query attention?
