# Transformer Block Architecture

**Category:** transformers
**Difficulty:** 3
**Tags:** architecture, fundamentals

## Question
Walk through the components of a standard transformer decoder block (as used in GPT).

## Answer
**Decoder Block (GPT-style):**

```
Input: x (shape: [batch, seq_len, d_model])

1. Pre-Norm + Self-Attention:
   norm_x = LayerNorm(x)
   attn_out = MultiHeadSelfAttention(norm_x)
   x = x + attn_out  # Residual connection

2. Pre-Norm + Feed-Forward:
   norm_x = LayerNorm(x)
   ff_out = FeedForward(norm_x)  # 2-layer MLP
   x = x + ff_out  # Residual connection

Output: x
```

**Feed-Forward Network:**
```
FFN(x) = GELU(xW_1 + b_1)W_2 + b_2

where:
  W_1: [d_model, 4*d_model]  # Expansion
  W_2: [4*d_model, d_model]  # Projection back
```

**Key Design Choices:**

- **Residual connections**: Enable gradient flow in deep networks (GPT-3 has 96 layers)
- **Pre-norm**: More stable than post-norm for deep networks
- **4x expansion**: FFN hidden dim = 4 × d_model is standard
- **GELU activation**: Smoother than ReLU, slightly better for LLMs

**Parameters per block** (d_model=768):
- Attention: ~2.4M parameters (QKV projections + output)
- FFN: ~4.7M parameters (most parameters!)
- LayerNorms: ~3K (negligible)

## Follow-up Questions
- Why is FFN hidden dim 4x the model dimension?
- What fraction of parameters are in attention vs FFN?
