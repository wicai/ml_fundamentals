# Softmax & Numerical Stability

**Category:** foundations
**Difficulty:** 3
**Tags:** fundamentals, numerical, attention

## Question
What is the softmax function and why do we subtract the max for numerical stability?

## Answer
**Softmax**: Convert logits to probability distribution.

**Formula:**
```
softmax(z)_i = exp(z_i) / Σ_j exp(z_j)
```

**Properties:**
1. Outputs sum to 1: Σ softmax(z)_i = 1
2. All outputs positive: softmax(z)_i > 0
3. Preserves order: z_i > z_j → softmax(z)_i > softmax(z)_j
4. Differentiable: Smooth gradients

**Numerical stability problem:**

**Overflow:**
```
z = [1000, 1001, 1002]
exp(1000) = 2.7 × 10^434  →  Infinity in float32!

softmax fails with inf / inf = NaN
```

**Underflow:**
```
z = [-1000, -1001, -1002]
exp(-1000) ≈ 0 in float32

softmax = 0 / 0 = NaN
```

**Solution: Subtract max**
```
softmax(z)_i = exp(z_i - max(z)) / Σ_j exp(z_j - max(z))

Mathematically equivalent:
  exp(z_i - c) / Σ exp(z_j - c) = exp(z_i) / Σ exp(z_j)
  (c cancels out)
```

**Why it works:**
```
z = [1000, 1001, 1002]
max(z) = 1002
z - max(z) = [-2, -1, 0]

exp(-2) = 0.135
exp(-1) = 0.368
exp(0) = 1.0

All computable! Largest value = 1.0
```

**Implementation:**
```python
def softmax(z):
    # Numerical stability: subtract max
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
```

**Gradient of softmax:**
```
For softmax(z)_i:
  ∂softmax_i/∂z_j = softmax_i * (δ_ij - softmax_j)

where δ_ij = 1 if i==j else 0

Jacobian is non-diagonal (all logits affect all probabilities)
```

**Softmax with cross-entropy (fused):**

**Naive:**
```
1. p = softmax(z)
2. loss = -log(p_correct)
3. Backprop through both
```

**Fused (better):**
```
Combined gradient:
  ∂loss/∂z_i = p_i - 1[i == correct]

Much simpler! No need to store softmax values
Numerically stable
```

**Implementation:**
```python
def cross_entropy_with_softmax(logits, targets):
    # Shift for stability
    logits_max = np.max(logits, axis=-1, keepdims=True)
    logits_shifted = logits - logits_max

    # Log-sum-exp trick
    log_sum_exp = np.log(np.sum(np.exp(logits_shifted), axis=-1, keepdims=True))

    # Cross-entropy
    log_probs = logits_shifted - log_sum_exp
    loss = -log_probs[range(len(targets)), targets].mean()

    return loss
```

**LogSumExp trick:**
```
log(Σ exp(z_i)) = log(Σ exp(z_i - max(z))) + max(z)

Used for stable computation of log probabilities
```

**Softmax temperature:**
```
softmax(z / T)_i = exp(z_i / T) / Σ_j exp(z_j / T)

T > 1: Flatter distribution (more random)
T < 1: Sharper distribution (more confident)
T → 0: Approaches argmax (deterministic)
T → ∞: Approaches uniform
```

**Sparse softmax:**
```
Standard softmax: Dense (all values > 0)

Sparse alternatives:
  - Sparsemax: Can have exact zeros
  - Entmax: Generalization

Used in some attention mechanisms
```

**Attention softmax specifics:**

**Masked softmax:**
```
scores = QK^T
scores[mask] = -inf  # Future tokens in causal attention

softmax([-inf, 0.5, 1.0]) = [0, 0.38, 0.62]

-inf → exp(-inf) = 0 → masked out
```

**Flash Attention softmax:**
```
Compute softmax in blocks (online algorithm)
Avoid materializing full attention matrix
More complex but same result
```

## Follow-up Questions
- Why does subtracting max preserve softmax values?
- What's the gradient of softmax + cross-entropy?
- How does temperature affect the output distribution?
