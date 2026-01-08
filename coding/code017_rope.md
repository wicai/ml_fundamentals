# Implement RoPE (Rotary Positional Embeddings)

**Category:** coding
**Difficulty:** 4
**Tags:** coding, positional-encoding, llama, modern-architectures

## Question
Implement Rotary Positional Embeddings (RoPE) as used in Llama, GPT-NeoX, and other modern LLMs.

RoPE applies a rotation to query and key vectors based on their position. This encodes relative position information directly into the attention mechanism.

**Function signature:**
```python
def apply_rotary_pos_emb(x, position_ids, theta=10000.0):
    """
    Apply rotary positional embeddings to input tensor.

    Args:
        x: input tensor of shape (batch, seq_len, num_heads, head_dim)
        position_ids: position indices, shape (batch, seq_len) or (seq_len,)
        theta: base for frequency computation
    Returns:
        tensor with RoPE applied, same shape as x
    """
    pass
```

## Answer

**Key concepts:**
1. Split head_dim into pairs
2. Compute rotation frequencies for each pair
3. Apply 2D rotation based on position
4. Rotation matrix: [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]

**Reference implementation:**
```python
import torch
import math

def apply_rotary_pos_emb(x, position_ids, theta=10000.0):
    """
    Apply RoPE to input tensor.
    """
    # x: (batch, seq_len, num_heads, head_dim)
    batch_size, seq_len, num_heads, head_dim = x.shape

    # Ensure head_dim is even (required for pairing)
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # Handle position_ids shape
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

    # Compute frequency for each dimension pair
    # freq_i = 1 / (theta ^ (2i / head_dim)) for i in [0, head_dim/2)
    dim_indices = torch.arange(0, head_dim // 2, dtype=torch.float32, device=x.device)
    freqs = 1.0 / (theta ** (2 * dim_indices / head_dim))  # (head_dim // 2,)

    # Compute angles: position * frequency
    # (batch, seq_len, 1) * (head_dim // 2,) -> (batch, seq_len, head_dim // 2)
    angles = position_ids.unsqueeze(-1).float() * freqs.unsqueeze(0).unsqueeze(0)

    # Compute cos and sin
    cos_angles = torch.cos(angles)  # (batch, seq_len, head_dim // 2)
    sin_angles = torch.sin(angles)

    # Reshape x into pairs: (batch, seq_len, num_heads, head_dim // 2, 2)
    x_pairs = x.reshape(batch_size, seq_len, num_heads, head_dim // 2, 2)

    # Split into even and odd components
    x_even = x_pairs[..., 0]  # (batch, seq_len, num_heads, head_dim // 2)
    x_odd = x_pairs[..., 1]

    # Apply rotation
    # [cos, -sin]   [x_even]
    # [sin,  cos] * [x_odd ]
    cos_angles = cos_angles.unsqueeze(2)  # Add num_heads dimension
    sin_angles = sin_angles.unsqueeze(2)

    rotated_even = x_even * cos_angles - x_odd * sin_angles
    rotated_odd = x_even * sin_angles + x_odd * cos_angles

    # Combine back
    rotated_pairs = torch.stack([rotated_even, rotated_odd], dim=-1)

    # Reshape back to original shape
    output = rotated_pairs.reshape(batch_size, seq_len, num_heads, head_dim)

    return output

# Alternative: More efficient implementation
def precompute_freqs_cis(head_dim, max_seq_len, theta=10000.0):
    """
    Precompute cos and sin frequencies for RoPE.
    Returns complex numbers representing rotations.
    """
    # Compute frequencies
    dim_indices = torch.arange(0, head_dim // 2, dtype=torch.float32)
    freqs = 1.0 / (theta ** (2 * dim_indices / head_dim))

    # Compute angles for all positions
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)  # (max_seq_len, head_dim // 2)

    # Convert to complex numbers (cos + i*sin)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)  # e^(i*theta)

    return freqs_cis

def apply_rotary_pos_emb_complex(x, freqs_cis):
    """
    Apply RoPE using complex number representation (more efficient).

    Args:
        x: (batch, seq_len, num_heads, head_dim)
        freqs_cis: (seq_len, head_dim // 2) complex tensor
    """
    batch_size, seq_len, num_heads, head_dim = x.shape

    # Reshape to pairs and convert to complex
    x_pairs = x.reshape(batch_size, seq_len, num_heads, head_dim // 2, 2)
    x_complex = torch.view_as_complex(x_pairs.float())  # (batch, seq_len, num_heads, head_dim // 2)

    # Apply rotation (complex multiplication)
    freqs_cis = freqs_cis[None, :seq_len, None, :]  # Add batch and head dims
    x_rotated = x_complex * freqs_cis

    # Convert back to real
    x_out = torch.view_as_real(x_rotated).reshape(batch_size, seq_len, num_heads, head_dim)

    return x_out.type_as(x)
```

**Usage in attention:**
```python
class AttentionWithRoPE(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=2048):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Precompute RoPE frequencies
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(self.head_dim, max_seq_len)
        )

    def forward(self, x, position_ids=None):
        batch_size, seq_len, d_model = x.shape

        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE to Q and K (not V!)
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device)

        Q = apply_rotary_pos_emb_complex(Q, self.freqs_cis)
        K = apply_rotary_pos_emb_complex(K, self.freqs_cis)

        # Transpose for attention
        Q = Q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(output)

        return output
```

**Testing:**
```python
# Test basic RoPE
batch_size, seq_len, num_heads, head_dim = 2, 10, 8, 64
x = torch.randn(batch_size, seq_len, num_heads, head_dim)
position_ids = torch.arange(seq_len)

# Apply RoPE
x_rotated = apply_rotary_pos_emb(x, position_ids)

print(f"Input shape: {x.shape}")
print(f"Output shape: {x_rotated.shape}")
print(f"Values changed: {not torch.allclose(x, x_rotated)}")

# Test relative position property
# RoPE encodes relative positions: RoPE(q, m) · RoPE(k, n) depends on (m-n)
q_pos_0 = apply_rotary_pos_emb(x[:, 0:1], torch.tensor([0]))
k_pos_1 = apply_rotary_pos_emb(x[:, 1:2], torch.tensor([1]))

q_pos_5 = apply_rotary_pos_emb(x[:, 0:1], torch.tensor([5]))
k_pos_6 = apply_rotary_pos_emb(x[:, 1:2], torch.tensor([6]))

# These should have similar relationship (same relative distance)
dot_01 = (q_pos_0 * k_pos_1).sum()
dot_56 = (q_pos_5 * k_pos_6).sum()

print(f"\nRelative position property:")
print(f"Dot product (pos 0, 1): {dot_01:.4f}")
print(f"Dot product (pos 5, 6): {dot_56:.4f}")
print(f"Similar: {torch.allclose(dot_01, dot_56, rtol=0.1)}")

# Test full attention module
model = AttentionWithRoPE(d_model=512, num_heads=8)
x_input = torch.randn(2, 10, 512)
output = model(x_input)

print(f"\nAttention output shape: {output.shape}")
```

**Common mistakes:**
1. ❌ Applying RoPE to values (should only apply to Q and K)
2. ❌ Wrong dimension for rotation (need even head_dim)
3. ❌ Not handling relative position correctly
4. ❌ Wrong frequency computation formula

## Follow-up Questions
- Why apply RoPE to Q and K but not V?
- How does RoPE enable length extrapolation?
- What's the advantage over absolute positional encoding?
