# Residual Connections & Skip Connections

**Category:** foundations
**Difficulty:** 3
**Tags:** architecture, training, fundamentals

## Question
What are residual connections and why are they critical for deep networks?

## Answer
**Residual Connection (Skip Connection)**: Add input to output of a layer.

**Standard layer:**
```
y = F(x)
```

**With residual:**
```
y = F(x) + x
```

**Original motivation (ResNet, 2015):**
- Very deep networks (50+ layers) performed worse than shallow ones
- Not overfitting → optimization problem
- Residual connections solved this → ResNet-152 (2015)

## **Why They Work:**

**1. Gradient flow:**

**Without residual:**
```
∂L/∂x = ∂L/∂y * ∂y/∂x
       = ∂L/∂y * ∂F/∂x

If ∂F/∂x small → vanishing gradients
50 layers: (0.9)^50 = 0.005 (gradient almost zero)
```

**With residual:**
```
y = F(x) + x
∂y/∂x = ∂F/∂x + 1  # Identity path!

∂L/∂x = ∂L/∂y * (∂F/∂x + 1)
      = ∂L/∂y * ∂F/∂x + ∂L/∂y

Always have direct gradient path: ∂L/∂y
```

**2. Ensemble interpretation:**
```
Network with n residual blocks = ensemble of 2^n paths

Each path either uses or skips each block
All paths contribute to gradient

Dropout-like effect without dropout
```

**3. Easier optimization:**
```
Learning F(x) = 0 (identity) is trivial
Learning y = x directly is hard (initialization)

Residual: Learn the "delta" from identity
F(x) starts near 0, learns refinement
```

## **In Transformers:**

**Transformer block:**
```
x' = x + Attention(x)  # Residual 1
x'' = x' + FFN(x')     # Residual 2

Each sub-layer adds to input (not replaces)
```

**With LayerNorm (Pre-Norm):**
```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))

Modern standard (GPT-2+, LLaMA)
More stable than Post-Norm
```

**With LayerNorm (Post-Norm):**
```
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))

Original Transformer design
Less stable for very deep (100+ layers)
```

## **Variants:**

**Weighted residual:**
```
y = α*F(x) + x

α learned or fixed (α < 1 for stability)
Fixup initialization uses this
```

**Dense connections (DenseNet):**
```
y = F([x, x_1, x_2, ...])  # Concat all previous

More connections than ResNet
Not common in transformers (memory)
```

**Highway networks:**
```
g = sigmoid(W_g * x)  # Gate
y = g * F(x) + (1-g) * x

Learned gating, precursor to LSTM/GRU style
```

## **Depth & Residuals:**

**Without residuals:**
```
Practical limit: ~10-20 layers
VGG-19: Very deep for its time (2014)
```

**With residuals:**
```
ResNet-152: 152 layers (2015)
Transformers: 96 layers (GPT-3), 200+ layers possible
No depth limit in practice (with proper init/norm)
```

**Diminishing returns:**
```
Most gains from 10 → 50 layers
50 → 200 layers: Smaller gains, much more cost

LLaMA-7B: 32 layers
LLaMA-70B: 80 layers (not linear scaling)
```

## **Gradient magnitudes:**

**Empirical finding:**
```
Early layers: Small gradients (many hops from loss)
Late layers: Large gradients (close to loss)

Residuals help but don't fully equalize
Solution: Layer-wise learning rates or scaling
```

## **Initialization with residuals:**

**Naive:**
```
If F(x) and x have same scale → y = 2x (doubles scale)
After 50 layers: 2^50 × scale (explodes)
```

**Solutions:**

**1. Scaled initialization:**
```
Initialize F weights with factor 1/√L
where L = number of residual blocks

GPT uses this
```

**2. Learnable scaling (ReZero):**
```
y = x + α*F(x)
α initialized to 0, learned

F(x) contribution grows during training
```

**3. FixUp initialization:**
```
Careful initialization scheme
Can train deep networks without normalization
Less common (LayerNorm preferred)
```

## **In Vision (ResNet):**

**Bottleneck block:**
```
x → 1×1 conv (reduce dim)
  → 3×3 conv (process)
  → 1×1 conv (expand dim)
  → add x

Fewer params than naive 3×3 conv
```

**Downsampling:**
```
When spatial size changes or channels change:
  y = F(x) + W_s * x  # Projection shortcut

W_s: 1×1 conv to match dimensions
```

## **Common mistakes:**

**1. Forgetting residual:**
```
❌ y = Attention(x)  # No residual
✓ y = x + Attention(x)

Harder to train, worse performance
```

**2. Dimension mismatch:**
```
❌ y = x + F(x)  # If shapes don't match
✓ y = x + Projection(F(x))  # Or project x
```

**3. Too many residuals:**
```
Adding residual to every single operation is overkill
Standard: Residual per "block" (attention + FFN)
```

## **Modern best practices:**

**Transformers:**
```
Pre-Norm residual (standard)
No scaling (LayerNorm handles it)
```

**Vision:**
```
ResNet blocks (proven)
Sometimes + channel attention (SE blocks)
```

**Deep networks (100+ layers):**
```
Consider:
  - Pre-norm
  - Scaled initialization
  - Gradient clipping
  - Smaller learning rate
```

## Follow-up Questions
- How do residual connections prevent vanishing gradients?
- What's the difference between pre-norm and post-norm residual blocks?
- Why can't you train very deep networks without residuals?
