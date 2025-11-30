# Rotary Position Embeddings (RoPE)

**Category:** transformers
**Difficulty:** 4
**Tags:** position, architecture, attention

## Question
What is RoPE (Rotary Position Embeddings) and why is it used in modern LLMs?

## Answer
**RoPE**: Encode position information by rotating query and key vectors.

**Intuition**: Represent position as a rotation angle in 2D planes.

**Algorithm:**
```
For each head dimension pair (i, i+1):
  θ = pos / 10000^(2i / d_head)

  # Rotate q and k by position-dependent angle
  q'_i = q_i * cos(θ) - q_{i+1} * sin(θ)
  q'_{i+1} = q_i * sin(θ) + q_{i+1} * cos(θ)

  # Same for k
  k'_i = k_i * cos(θ) - k_{i+1} * sin(θ)
  k'_{i+1} = k_i * sin(θ) + k_{i+1} * cos(θ)
```

**Key property**:
```
q_{pos_i}^T k_{pos_j} only depends on relative distance (i - j)

Not absolute positions i, j!
```

**Why RoPE is great:**

1. **Relative positioning**: Attention scores depend on relative distance
2. **Extrapolation**: Can handle sequences longer than training
   - Trained on 2K → inference on 8K works reasonably
3. **No learned parameters**: Purely geometric, no additional params
4. **Efficient**: Applied element-wise, no extra memory

**vs Other position encodings:**

| Method | Relative? | Extrapolation? | Learned? |
|--------|-----------|----------------|----------|
| Sinusoidal | No | Yes | No |
| Learned absolute | No | No | Yes |
| RoPE | Yes | Yes | No |
| ALiBi | Yes | Excellent | No |

**Adoption:**
- **LLaMA 1 & 2**: RoPE
- **GPT-NeoX**: RoPE
- **PaLM**: RoPE
- Becoming standard for decoder-only LLMs

**Length extrapolation:**
- Works but degrades beyond ~2× training length
- Tricks: YaRN, CodeLLaMA extended context
- Better than learned absolute embeddings

**Implementation detail:**
```python
def apply_rope(q, k, positions):
    # Precompute frequencies
    freqs = 1.0 / (10000 ** (torch.arange(0, d, 2) / d))
    # Compute angles for each position
    angles = positions[:, None] * freqs[None, :]
    # Apply rotation (complex number representation)
    q_rot = apply_rotation(q, cos(angles), sin(angles))
    k_rot = apply_rotation(k, cos(angles), sin(angles))
    return q_rot, k_rot
```

## Follow-up Questions
- How does RoPE achieve relative position encoding?
- Why can RoPE extrapolate to longer sequences?
- How does RoPE compare to ALiBi?
