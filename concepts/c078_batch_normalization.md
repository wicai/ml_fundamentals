# Batch Normalization vs Layer Normalization

**Category:** foundations
**Difficulty:** 3
**Tags:** normalization, architecture, training

## Question
What's the difference between batch normalization and layer normalization? Why do transformers use layer norm?

## What to Cover
- **Set context by**: Explaining normalization stabilizes training by standardizing activations
- **Must mention**: The dimension difference (BN across batch, LN across features), why transformers use LN (batch independence, variable sequences, inference with batch=1)
- **Show depth by**: Mentioning RMSNorm (LLaMA, faster), pre-norm vs post-norm placement, and other variants (GroupNorm, InstanceNorm)
- **Avoid**: Only describing formulas without explaining why transformers specifically need layer norm

## Answer
**Normalization**: Standardize activations to stabilize training.

## **Batch Normalization (BatchNorm)**

**Operation:**
```
For each feature across batch:
  μ = mean(x[:, i])  # Mean across batch dimension
  σ = std(x[:, i])   # Std across batch dimension

  x̂[:, i] = (x[:, i] - μ) / (σ + ε)
  y[:, i] = γ * x̂[:, i] + β  # Learnable scale/shift
```

**Dimensions:**
```
Input: [batch, features]
Normalize across batch dimension for each feature

For CNNs [batch, channels, height, width]:
  Normalize across (batch, height, width) for each channel
```

**Training vs inference:**
```
Training: Use batch statistics (μ, σ computed from current batch)
Inference: Use running average of μ, σ from training
  (Otherwise single example would have σ=0!)
```

**Properties:**
- ✓ Stabilizes training (less sensitive to initialization, LR)
- ✓ Acts as regularizer (noise from batch sampling)
- ✓ Allows higher learning rates
- ✗ Depends on batch size (small batch → noisy estimates)
- ✗ Different behavior train vs test
- ✗ Doesn't work well with batch size = 1

## **Layer Normalization (LayerNorm)**

**Operation:**
```
For each example across features:
  μ = mean(x[i, :])  # Mean across feature dimension
  σ = std(x[i, :])   # Std across feature dimension

  x̂[i, :] = (x[i, :] - μ) / (σ + ε)
  y[i, :] = γ * x̂[i, :] + β  # Learnable scale/shift
```

**Dimensions:**
```
Input: [batch, features]
Normalize across feature dimension for each example

For transformers [batch, seq_len, d_model]:
  Normalize across d_model for each (batch, seq_len) position
```

**Properties:**
- ✓ Batch size independent (works with batch=1)
- ✓ Same behavior train vs test (no running stats)
- ✓ Works with variable sequence lengths
- ✗ No regularization effect from batch noise

## **Why Transformers Use Layer Norm:**

**1. Batch size independence:**
```
LLM inference: batch size = 1 (single user query)
BatchNorm would fail (can't compute statistics)
LayerNorm works fine
```

**2. Variable sequence lengths:**
```
Batch: [
  sequence 1: length 50,
  sequence 2: length 200,
  sequence 3: length 100
]

BatchNorm would need padding → waste
LayerNorm handles each independently
```

**3. Recurrence / autoregressive:**
```
Generating token by token → batch size = 1
Need same normalization scheme train and inference
```

**4. Simpler:**
```
No running statistics to track
No train/test mode switching
```

## **Other Normalization Types:**

**RMSNorm (Root Mean Square Norm):**
```
Used in: LLaMA, GPT-NeoX

Simplification of LayerNorm:
  x̂ = x / RMS(x)
  y = γ * x̂

No mean subtraction, no bias
Slightly faster, similar performance
```

**GroupNorm:**
```
Split channels into groups, normalize within group

Between LayerNorm and InstanceNorm
Used in some vision models
```

**InstanceNorm:**
```
Normalize each channel independently
Used in style transfer (images)
```

**Comparison:**

| Method | Normalize across | Train=Test | Batch size | Use case |
|--------|------------------|------------|------------|----------|
| BatchNorm | Batch + spatial | No | Large | CNNs |
| LayerNorm | Features | Yes | Any | Transformers, RNNs |
| GroupNorm | Groups | Yes | Any | Vision (small batch) |
| InstanceNorm | Spatial | Yes | Any | Style transfer |
| RMSNorm | Features | Yes | Any | LLMs (faster) |

## **Where LayerNorm is applied in transformers:**

**Pre-Norm (modern):**
```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))

LayerNorm before each sub-layer
More stable for deep networks (GPT-2+)
```

**Post-Norm (original):**
```
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))

LayerNorm after residual addition
Original Transformer design
Less stable for very deep networks
```

## **Implementation details:**

**Epsilon (ε):**
```
Prevents division by zero: σ + ε
Typical: ε = 1e-5 or 1e-6
```

**Learnable parameters:**
```
γ (scale): Initialize to 1
β (bias): Initialize to 0

Allows network to learn to undo normalization if needed
```

**Gradient flow:**
```
LayerNorm helps gradient flow:
  Prevents activations from exploding/vanishing
  Centered, unit variance → stable gradients
```

## Follow-up Questions
- Why doesn't BatchNorm work well with small batches?
- What's the difference between pre-norm and post-norm?
- Why is RMSNorm faster than LayerNorm?
