# Gradient Clipping

**Category:** training
**Difficulty:** 2
**Tags:** training, optimization, stability

## Question
What is gradient clipping and why is it important for training transformers?

## Answer
**Gradient Clipping**: Limit gradient magnitude to prevent instability.

**Two types:**

**1. Clip by value:**
```
g = clip(g, -threshold, threshold)
```
Rarely used, too crude.

**2. Clip by global norm (standard):**
```
global_norm = sqrt(Σ ||g_i||²) for all gradients
if global_norm > threshold:
    g = g * (threshold / global_norm)
```

**Why needed for transformers:**

1. **Exploding gradients**: Rare but catastrophic when they occur
2. **Deep networks**: 96 layers (GPT-3) → gradient can compound
3. **Attention**: Softmax can produce extreme values with bad initialization
4. **Loss spikes**: Without clipping, single bad batch can diverge training

**Typical threshold: 1.0**
- GPT-3, LLaMA: 1.0
- BERT: 1.0
- Some use 5.0 or higher

**How it helps:**

- **Stability**: Prevents NaN/Inf from gradient explosion
- **Recovery**: Can recover from bad batches instead of diverging
- **Insurance**: Doesn't hurt if gradients are normal, saves you if they spike

**Trade-off**: If clipping activates frequently, you have deeper issues (LR too high, data problem, numerical instability).

**Monitoring**: Log `global_norm` and `fraction_clipped` to diagnose issues.

## Follow-up Questions
- How does this interact with gradient accumulation?
- When would high clipping frequency indicate a problem?
- Does gradient clipping change the optimization landscape?
