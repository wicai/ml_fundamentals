# LoRA & Parameter-Efficient Fine-Tuning

**Category:** training
**Difficulty:** 4
**Tags:** finetuning, efficiency, optimization

## Question
What is LoRA and why is it more efficient than full fine-tuning?

## Answer
**LoRA (Low-Rank Adaptation)**: Fine-tune by adding small trainable matrices instead of updating all weights.

**Standard fine-tuning:**
```
W ∈ R^{d×k} (pretrained)
During fine-tuning: Update entire W
Trainable parameters: d × k
```

**LoRA:**
```
Keep W frozen
Add: W' = W + B*A

where:
  B ∈ R^{d×r} (trainable)
  A ∈ R^{r×k} (trainable)
  r << min(d, k) (rank, typically 8-64)

Trainable parameters: r*(d+k) << d*k
```

**Example:**
```
Attention layer: d=4096, k=4096
Full FT: 16.7M parameters
LoRA r=16: 131K parameters (128× smaller!)

Savings: 99.2% fewer parameters
```

**Why it works:**

**Hypothesis**: Fine-tuning updates lie in low-rank subspace
- Full weight change ΔW has low intrinsic dimensionality
- Can approximate with low-rank decomposition ΔW ≈ B*A

**Benefits:**

1. **Memory**: 10-100× less memory (only train B, A)
2. **Speed**: Faster training (fewer parameters)
3. **Storage**: One base model + multiple small LoRA adapters
   - Base: 7GB
   - LoRA adapter: 50MB
   - Can swap adapters at runtime!

4. **Multi-tenant**: Serve many custom models from one base

**Where to apply LoRA:**

Most effective on:
- ✓ Attention: Q, K, V, O projections (standard)
- ✓ Feed-forward: W_1, W_2
- ✗ Embeddings: Usually frozen
- ✗ Layer norm: Frozen

**Hyperparameters:**

- **Rank (r)**: 8-64 typical
  - Higher = more capacity, more memory
  - 16 is sweet spot for most tasks

- **Alpha (scaling)**: Often 2*r
  - Scales the LoRA contribution: W + (alpha/r)*B*A

- **Target modules**: Which layers get LoRA
  - Q,V is common
  - Q,K,V,O for more capacity

**Variants:**

**QLoRA:**
- LoRA + 4-bit quantization of base model
- Fine-tune 70B model on single GPU!

**AdaLoRA:**
- Adaptive rank allocation (important layers get higher rank)

**DoRA:**
- Decomposes magnitude and direction
- Slightly better than LoRA

**Limitations:**

- Slightly worse than full fine-tuning (small gap)
- Not as effective for very different domains
- Rank selection can be tricky

**Inference:**
```
Option 1: Keep separate (dynamic adapter swapping)
  y = (W + B*A)x = Wx + BAx

Option 2: Merge into base model
  W_merged = W + B*A
  (No runtime overhead)
```

## Follow-up Questions
- How do you choose the rank r?
- Can you merge multiple LoRA adapters?
- Why does low-rank approximation work for fine-tuning?
