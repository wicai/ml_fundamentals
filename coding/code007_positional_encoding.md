# Implement Sinusoidal Positional Encoding

**Category:** coding
**Difficulty:** 3
**Tags:** coding, transformers, positional-encoding

## Question
Implement sinusoidal positional encoding as used in the original Transformer paper.

Your implementation should:
- Generate position encodings for any sequence length
- Use sine for even dimensions, cosine for odd dimensions
- Support different model dimensions
- Be deterministic (not learned)

**Function signature:**
```python
import torch

def get_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """
    Generate sinusoidal positional encodings.

    Args:
        seq_len: sequence length
        d_model: model dimension (embedding size)
    Returns:
        positional encoding tensor of shape (seq_len, d_model)
    """
    pass
```

## Answer

**Key concepts:**
1. Position encoding formula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
2. PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
3. Different frequencies for different dimensions

**Reference implementation:**
```python
import torch
import math

def get_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    # Initialize positional encoding matrix
    pe = torch.zeros(seq_len, d_model)

    # Create position indices [0, 1, 2, ..., seq_len-1]
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)

    # Create dimension indices [0, 2, 4, ..., d_model-2]
    # These represent the pairs of dimensions (2i, 2i+1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) *
        (-math.log(10000.0) / d_model)
    )  # (d_model // 2,)

    # Apply sin to even dimensions (0, 2, 4, ...)
    pe[:, 0::2] = torch.sin(position * div_term)

    # Apply cos to odd dimensions (1, 3, 5, ...)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

# Alternative implementation (more explicit)
def get_positional_encoding_v2(seq_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(seq_len, d_model)

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            # Compute the denominator
            denominator = 10000 ** (i / d_model)

            # Apply sin to even indices
            pe[pos, i] = math.sin(pos / denominator)

            # Apply cos to odd indices (if exists)
            if i + 1 < d_model:
                pe[pos, i + 1] = math.cos(pos / denominator)

    return pe
```

**As a Module:**
```python
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        # Precompute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x
```

**Testing:**
```python
# Test
seq_len, d_model = 100, 512
pe = get_positional_encoding(seq_len, d_model)

print(f"Shape: {pe.shape}")  # (100, 512)
print(f"Range: [{pe.min():.2f}, {pe.max():.2f}]")  # Should be [-1, 1]

# Visualize pattern (first few positions and dimensions)
print("\nFirst 5 positions, first 8 dimensions:")
print(pe[:5, :8])

# Check that nearby positions have similar encodings
pos_0 = pe[0]
pos_1 = pe[1]
pos_50 = pe[50]

similarity_01 = torch.cosine_similarity(pos_0.unsqueeze(0), pos_1.unsqueeze(0))
similarity_050 = torch.cosine_similarity(pos_0.unsqueeze(0), pos_50.unsqueeze(0))

print(f"\nSimilarity(pos_0, pos_1): {similarity_01.item():.4f}")
print(f"Similarity(pos_0, pos_50): {similarity_050.item():.4f}")
print("Nearby positions should be more similar!")

# Test module
x = torch.randn(2, 100, 512)
pos_enc = PositionalEncoding(d_model=512)
x_encoded = pos_enc(x)
print(f"\nWith module - output shape: {x_encoded.shape}")
```

**Common mistakes:**
1. ❌ Using `math.log(10000)` instead of `math.log(10000.0)` (integer division)
2. ❌ Wrong indexing for sin/cos (not alternating properly)
3. ❌ Forgetting to unsqueeze position for broadcasting
4. ❌ Not handling odd d_model correctly

## Follow-up Questions
- Why use sin/cos instead of learned embeddings?
- Why different frequencies for different dimensions?
- How does this enable the model to learn relative positions?
