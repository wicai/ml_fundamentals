# Activation Functions

**Category:** foundations
**Difficulty:** 2
**Tags:** activation, architecture, fundamentals

## Question
What activation functions are used in transformers and why GELU instead of ReLU?

## What to Cover
- **Set context by**: Comparing the common activations (ReLU, GELU, SiLU/Swish)
- **Must mention**: Why GELU is preferred (smoothness, probabilistic interpretation, empirically better), where activations are applied in transformers (FFN only)
- **Show depth by**: Mentioning gated variants (GLU, SwiGLU) used in modern models like LLaMA
- **Avoid**: Only describing ReLU without explaining why transformers moved to GELU/SiLU

## Answer
**Common activations:**

**ReLU** (Rectified Linear Unit):
```
ReLU(x) = max(0, x)

Pros: Simple, no vanishing gradient
Cons: "Dying ReLU" (neurons stuck at 0)
```

**GELU** (Gaussian Error Linear Unit):
```
GELU(x) ≈ x * Φ(x)  # Φ = CDF of standard normal

Approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715*x³)))

Smooth, non-monotonic
```

**Why transformers use GELU:**

1. **Smoothness**: Continuously differentiable (vs ReLU's kink)
2. **Probabilistic**: Stochastic regularization interpretation
3. **Empirical**: Works better for language models
4. **Non-monotonic**: Can have negative values (richer representations)

**SiLU / Swish** (alternative to GELU):
```
SiLU(x) = x * sigmoid(x)

Used in some models (e.g., LLaMA uses SiLU in FFN)
Very similar to GELU in practice
```

**Comparison:**
```
x = -2  →  ReLU: 0,    GELU: -0.04,  SiLU: -0.24
x = -1  →  ReLU: 0,    GELU: -0.16,  SiLU: -0.27
x = 0   →  ReLU: 0,    GELU: 0,      SiLU: 0
x = 1   →  ReLU: 1,    GELU: 0.84,   SiLU: 0.73
x = 2   →  ReLU: 2,    GELU: 1.96,   SiLU: 1.76
```

**Where used in transformers:**

- **Feed-forward network**: GELU/SiLU after first linear layer
- **Attention**: No activation (just softmax)
- **Output**: No activation (logits → softmax later)

**GLU variants** (Gated Linear Units):
```
Standard FFN: GELU(xW_1)W_2

GLU: (x @ W_gate * sigmoid(x @ W_gate)) @ W_2
GeGLU: (x @ W_gate * GELU(x @ W_gate)) @ W_2
SwiGLU: (x @ W_gate * SiLU(x @ W_gate)) @ W_2
```

**SwiGLU** (LLaMA, PaLM):
- Gated version of SiLU
- Slightly better than standard GELU
- 50% more parameters (two weight matrices instead of one)

**Modern trend:** GELU/SiLU standard, SwiGLU for high-end models.

## Follow-up Questions
- Why is GELU better than ReLU for transformers?
- What's the computational cost difference?
- What is a gated activation (GLU)?
