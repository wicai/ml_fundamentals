# AdamW Optimizer

**Category:** training
**Difficulty:** 4
**Tags:** optimization, training, fundamentals

## Question
Why is AdamW the standard optimizer for LLMs? How does it differ from Adam?

## What to Cover
- **Set context by**: Briefly explaining adaptive learning rate optimizers vs SGD
- **Must mention**: The key difference (decoupled weight decay), typical hyperparameters (β1=0.9, β2=0.95), memory overhead (2x for m and v)
- **Show depth by**: Explaining why L2 regularization ≠ weight decay in Adam (the core insight)
- **Avoid**: Only reciting the Adam formulas without explaining the AdamW correction

## Answer
**Adam** (original):
```
m_t = β_1 * m_{t-1} + (1-β_1) * g_t        # 1st moment (momentum)
v_t = β_2 * v_{t-1} + (1-β_2) * g_t²       # 2nd moment (variance)
m_hat = m_t / (1 - β_1^t)                   # Bias correction
v_hat = v_t / (1 - β_2^t)
θ_t = θ_{t-1} - α * m_hat / (√v_hat + ε)   # Update
θ_t = θ_t - λ * θ_t                         # Weight decay (wrong!)
```

**AdamW** (corrected):
```
[Same momentum and variance updates]
θ_t = θ_{t-1} - α * m_hat / (√v_hat + ε)   # Update
θ_t = θ_t - λ * θ_{t-1}                     # Weight decay (correct!)
```

**Key difference**: Weight decay is applied to **original weights**, not **gradient-scaled update**.

**Why AdamW is better:**

1. **Decoupled weight decay**: L2 regularization and weight decay are NOT the same in Adam
2. **Better generalization**: Regularization actually works properly
3. **Hyperparameter independence**: Weight decay effect doesn't depend on gradient scale

**Typical hyperparameters (LLMs):**
- Learning rate: 3e-4 to 6e-4
- β_1: 0.9
- β_2: 0.95 or 0.999
- Weight decay: 0.1
- ε: 1e-8

**Why Adam family (vs SGD)?**
- **Adaptive learning rates**: Different learning rate per parameter
- **Less sensitive to LR**: Scales gradients by their historical variance
- **Sparse gradients**: Works well with embeddings (many zeros)

**Trade-off**: More memory (stores m and v for each parameter = 2x optimizer state).

## Follow-up Questions
- What's the difference between L2 regularization and weight decay in Adam?
- Why are β values typically 0.9 and 0.95?
- When would you use SGD instead of AdamW?
