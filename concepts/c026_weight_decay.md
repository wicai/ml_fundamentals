# Weight Decay / L2 Regularization

**Category:** foundations
**Difficulty:** 3
**Tags:** regularization, optimization, training

## Question
What is weight decay and how does it differ from L2 regularization in Adam/AdamW?

## Answer
**L2 Regularization** (traditional):
```
Loss = Task_loss + λ * Σ w_i²
Gradient: ∂L/∂w = ∂Task_loss/∂w + 2λw
```

**Weight Decay** (direct):
```
w_t = w_{t-1} - lr * (∂Task_loss/∂w) - lr * λ * w_{t-1}
```

**For SGD**: These are mathematically equivalent!

**For Adam**: They are NOT equivalent!

**L2 in Adam** (wrong):
- Gradient includes 2λw term
- Gets normalized by adaptive learning rates
- Effective regularization depends on gradient magnitude
- Doesn't work well in practice

**Weight Decay in Adam** (AdamW, correct):
- Directly decay weights: w = (1 - lr*λ) * w
- Independent of gradient
- Consistent regularization across parameters

**Why this matters:**
```
# Adam with L2
grad = compute_grad(w) + 2*lambda*w
m, v = update_moments(grad)
w = w - lr * m / sqrt(v)

# AdamW (correct)
grad = compute_grad(w)  # No L2 term!
m, v = update_moments(grad)
w = w - lr * m / sqrt(v)
w = w * (1 - lr * lambda)  # Separate weight decay
```

**Typical values:**
- λ = 0.01-0.1 for LLMs
- λ = 0.0001-0.001 for smaller models

**Effect:**
- Prevents weights from growing too large
- Improves generalization
- Biases toward simpler solutions

**Modern practice**: Always use AdamW, not Adam, if you want weight decay.

## Follow-up Questions
- Why does L2 regularization fail with adaptive learning rates?
- Should you apply weight decay to biases and layer norm parameters?
- How do you choose the weight decay coefficient?
