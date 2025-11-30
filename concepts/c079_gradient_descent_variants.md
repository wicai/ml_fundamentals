# Gradient Descent Variants: SGD vs Adam vs AdamW

**Category:** foundations
**Difficulty:** 3
**Tags:** optimization, training, fundamentals

## Question
Compare SGD, Adam, and AdamW optimizers. When should you use each?

## Answer
**Goal**: Update parameters to minimize loss.

## **SGD (Stochastic Gradient Descent)**

**Basic update:**
```
θ = θ - lr * ∇L(θ)
```

**With momentum:**
```
v_t = β * v_{t-1} + ∇L(θ)  # Velocity
θ = θ - lr * v_t

β typically 0.9
Accumulates gradients → faster convergence
```

**Properties:**
- Simple, well-understood
- Low memory (no extra state)
- Sensitive to learning rate
- Slow on problems with different scales

**When to use:**
✓ Computer vision (CNNs)
✓ When you have good learning rate schedule
✓ Want simplicity and less memory

✗ NLP / transformers (Adam better)

## **Adam (Adaptive Moment Estimation)**

**Algorithm:**
```
# First moment (momentum)
m_t = β_1 * m_{t-1} + (1-β_1) * g_t

# Second moment (variance)
v_t = β_2 * v_{t-1} + (1-β_2) * g_t²

# Bias correction
m̂_t = m_t / (1 - β_1^t)
v̂_t = v_t / (1 - β_2^t)

# Update
θ = θ - lr * m̂_t / (√v̂_t + ε)

Typical: β_1=0.9, β_2=0.999, ε=1e-8
```

**Key idea**: Adaptive learning rates per parameter
```
Parameters with large gradients → smaller effective LR
Parameters with small gradients → larger effective LR

Auto-scales learning rate based on gradient history
```

**Properties:**
- Adaptive per-parameter learning rates
- Less sensitive to learning rate choice
- More memory (stores m and v for each param)
- Works well out of the box

**Memory cost:**
```
Model: P parameters
Adam: 2P extra (m and v)
Total: 3P memory

For 7B model: 21B values to store!
```

**When to use:**
✓ NLP / transformers (standard)
✓ When LR tuning is hard
✓ Different parameter scales (embeddings vs weights)

## **AdamW (Adam with decoupled Weight Decay)**

**Key difference from Adam:**
```
Adam with L2 regularization (wrong):
  g_t = ∇L(θ) + λ*θ  # Add L2 to gradient
  [Adam update using g_t]

AdamW (correct):
  [Adam update using just ∇L(θ)]
  θ = θ - lr * λ * θ  # Separate weight decay step
```

**Why AdamW is better:**
```
In Adam: L2 penalty gets normalized by adaptive LR
  → Effective regularization depends on gradient magnitude

In AdamW: Weight decay independent of gradient
  → Consistent regularization across parameters
```

**Algorithm:**
```
m_t = β_1 * m_{t-1} + (1-β_1) * g_t
v_t = β_2 * v_{t-1} + (1-β_2) * g_t²
m̂_t = m_t / (1 - β_1^t)
v̂_t = v_t / (1 - β_2^t)

θ = θ - lr * m̂_t / (√v̂_t + ε)  # Adam update
θ = θ - lr * λ * θ  # Weight decay (new)

Typical: λ = 0.01-0.1
```

**When to use:**
✓ Almost always (if using Adam)
✓ Modern LLM training (standard)
✓ When you want weight decay / regularization

## **Comparison:**

| Aspect | SGD+Momentum | Adam | AdamW |
|--------|--------------|------|-------|
| Memory | 1× params | 3× params | 3× params |
| LR sensitivity | High | Low | Low |
| Vision (CNNs) | Good | Good | Good |
| NLP (Transformers) | Okay | Good | Best |
| Regularization | Separate | Coupled | Decoupled |
| Typical LR | 0.1-0.01 | 1e-3 to 1e-4 | 3e-4 to 6e-4 |

## **LLM Training Standard (2024):**

**AdamW with:**
```
lr: 3e-4 to 6e-4
β_1: 0.9
β_2: 0.95 or 0.999
weight_decay: 0.1
gradient_clipping: 1.0

Warmup: 2000 steps or 1% of training
Schedule: Cosine decay to min_lr (10% of max_lr)
```

## **Other optimizers:**

**Lion (Google, 2023):**
```
Update based on sign of momentum
Less memory than Adam (1× vs 3×)
Competitive performance

Still experimental
```

**Adafactor:**
```
Reduce memory by factorizing second moment
For very large models
More complex, less widely used
```

**LAMB (Layer-wise Adaptive Moments):**
```
Adam variant for large-batch training
Scaling to batch size 64K+
Used in some BERT training
```

## **Choosing optimizer:**

**Default:**
- Vision: SGD + momentum or AdamW
- NLP/LLMs: AdamW

**Budget:**
- Low memory: SGD
- Standard: AdamW

**Scale:**
- Small model: Either works
- Large model (70B+): AdamW standard

## **Learning rate tips:**

**SGD:**
```
Higher LR needed: 0.1-0.01
Sensitive to LR → need tuning

Warmup + step/cosine decay
```

**Adam/AdamW:**
```
Lower LR: 1e-4 to 1e-3
Less sensitive → works in wider range

Still benefit from warmup + decay
```

**Rule of thumb:**
```
If switching SGD → Adam: lr_adam = lr_sgd / 10
If switching Adam → AdamW: Same LR, add weight_decay
```

## Follow-up Questions
- Why is Adam better than SGD for transformers?
- What's the difference between L2 regularization and weight decay in Adam?
- How much memory overhead does Adam have?
