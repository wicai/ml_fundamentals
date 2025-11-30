# Layer Normalization

**Category:** transformers
**Difficulty:** 4
**Tags:** normalization, training, architecture

## Question
Why do transformers use Layer Norm instead of Batch Norm? What's the difference?

## Answer
**Batch Norm** (CNNs):
- Normalizes across batch dimension for each feature
- Formula: normalize(X[b, :, h, w]) across batch b

**Layer Norm** (Transformers):
- Normalizes across feature dimension for each example
- Formula: normalize(X[b, :]) across features

**Why Layer Norm for Transformers:**

1. **Batch size independence**: LN works with batch=1, BN needs large batches
2. **Sequence length variability**: Different sequence lengths in batch break BN
3. **Autoregressive generation**: At inference, generate one token at a time (batch=1)
4. **Recurrent stability**: Applying same normalization at each position

**Formula:**
```
y = γ * (x - μ) / (σ + ε) + β

where:
  μ = mean across features (dim d_model)
  σ = std across features
  γ, β = learned scale and shift parameters
```

**Pre-Norm vs Post-Norm:**
- Post-Norm: LayerNorm(x + Sublayer(x)) - original
- Pre-Norm: x + Sublayer(LayerNorm(x)) - modern default, more stable

## Follow-up Questions
- What's the gradient flow difference between pre-norm and post-norm?
- Why does pre-norm help with deep networks?
