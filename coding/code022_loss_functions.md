# Implement Loss Functions and Probability Transformations

**Category:** coding
**Difficulty:** 3
**Tags:** coding, loss functions, cross entropy, logits, numerical stability, KL divergence

## Question
Implement common loss functions and probability transformations from scratch, with proper numerical stability.

Your implementation should include:

1. **Transformations:**
   - `logits_to_probs`: Convert logits to probabilities using softmax
   - `probs_to_logprobs`: Convert probabilities to log-probabilities
   - `logits_to_logprobs`: Convert logits directly to log-probabilities (log-softmax)

2. **Loss Functions:**
   - `cross_entropy_loss`: Categorical cross-entropy from probabilities
   - `cross_entropy_from_logits`: Cross-entropy directly from logits (more stable)
   - `binary_cross_entropy`: Binary cross-entropy loss
   - `kl_divergence`: KL divergence between two distributions

3. **Numerical Stability:**
   - Handle numerical underflow/overflow correctly
   - Use log-sum-exp trick where appropriate
   - Avoid computing `log(0)` or `exp(large_number)`

**Function signatures:**
```python
def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities using softmax.

    Args:
        logits: shape (batch_size, num_classes) or (num_classes,)
    Returns:
        probs: same shape as logits, values in [0, 1], sum to 1
    """
    pass

def logits_to_logprobs(logits: np.ndarray) -> np.ndarray:
    """Convert logits to log-probabilities using log-softmax.

    More numerically stable than log(softmax(logits)).

    Args:
        logits: shape (batch_size, num_classes) or (num_classes,)
    Returns:
        log_probs: same shape as logits
    """
    pass

def cross_entropy_loss(probs: np.ndarray, targets: np.ndarray) -> float:
    """Compute categorical cross-entropy from probabilities.

    Args:
        probs: predicted probabilities, shape (batch_size, num_classes)
        targets: true class indices, shape (batch_size,)
    Returns:
        loss: scalar cross-entropy loss
    """
    pass

def cross_entropy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    """Compute cross-entropy directly from logits (numerically stable).

    Args:
        logits: raw model outputs, shape (batch_size, num_classes)
        targets: true class indices, shape (batch_size,)
    Returns:
        loss: scalar cross-entropy loss
    """
    pass

def binary_cross_entropy(probs: np.ndarray, targets: np.ndarray) -> float:
    """Compute binary cross-entropy loss.

    Args:
        probs: predicted probabilities, shape (batch_size,), values in [0, 1]
        targets: binary labels, shape (batch_size,), values in {0, 1}
    Returns:
        loss: scalar BCE loss
    """
    pass

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence KL(p || q) = sum(p * log(p/q)).

    Args:
        p: true distribution, shape (num_classes,)
        q: approximate distribution, shape (num_classes,)
    Returns:
        kl: scalar KL divergence (always >= 0)
    """
    pass
```

## Answer

**Key Concepts:**

1. **Logits vs Probabilities:**
   - Logits: raw, unnormalized scores (can be any real number)
   - Probabilities: normalized, sum to 1, in range [0, 1]

2. **Why Numerical Stability Matters:**
   - `exp(large_number)` → overflow (infinity)
   - `log(small_number)` → underflow (-infinity)
   - `log(0)` → undefined (-infinity)

3. **Log-Sum-Exp Trick:**
   ```
   log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
   ```
   Subtracting max prevents overflow in exp().

**Reference implementation:**
```python
import numpy as np
from typing import Union

def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    """
    Convert logits to probabilities using softmax.

    Softmax: p_i = exp(logit_i) / sum(exp(logit_j))

    Numerically stable version: subtract max before exp.
    """
    # Handle both 1D and 2D inputs
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    # Subtract max for numerical stability
    # This doesn't change the result but prevents overflow
    logits_max = np.max(logits, axis=1, keepdims=True)
    logits_shifted = logits - logits_max

    # Compute softmax
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    if squeeze:
        probs = probs.squeeze(0)

    return probs


def logits_to_logprobs(logits: np.ndarray) -> np.ndarray:
    """
    Convert logits to log-probabilities using log-softmax.

    Log-softmax: log_p_i = logit_i - log(sum(exp(logit_j)))

    This is more stable than log(softmax(logits)).
    """
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    # Use log-sum-exp trick
    logits_max = np.max(logits, axis=1, keepdims=True)
    logits_shifted = logits - logits_max

    # log(sum(exp(x))) = log(sum(exp(x - max))) + max
    log_sum_exp = np.log(np.sum(np.exp(logits_shifted), axis=1, keepdims=True)) + logits_max

    log_probs = logits - log_sum_exp

    if squeeze:
        log_probs = log_probs.squeeze(0)

    return log_probs


def cross_entropy_loss(probs: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute categorical cross-entropy from probabilities.

    CE = -1/N * sum(log(p[i, target[i]]))

    Warning: This is less numerically stable than computing from logits!
    """
    batch_size = probs.shape[0]

    # Add small epsilon to prevent log(0)
    eps = 1e-15
    probs_clipped = np.clip(probs, eps, 1 - eps)

    # Extract probabilities for the true classes
    # probs[i, targets[i]] for each i
    true_class_probs = probs_clipped[np.arange(batch_size), targets]

    # Cross-entropy: -log(p(true_class))
    log_probs = np.log(true_class_probs)
    loss = -np.mean(log_probs)

    return loss


def cross_entropy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute cross-entropy directly from logits (numerically stable).

    This is the preferred way to compute cross-entropy!

    CE = -1/N * sum(logit[i, target[i]] - log_sum_exp(logits[i]))
    """
    batch_size = logits.shape[0]

    # Compute log-probabilities using stable log-softmax
    log_probs = logits_to_logprobs(logits)

    # Extract log-probs for true classes
    true_class_log_probs = log_probs[np.arange(batch_size), targets]

    # Cross-entropy
    loss = -np.mean(true_class_log_probs)

    return loss


def binary_cross_entropy(probs: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute binary cross-entropy loss.

    BCE = -1/N * sum(y*log(p) + (1-y)*log(1-p))

    where y ∈ {0, 1} and p ∈ [0, 1]
    """
    # Clip to prevent log(0)
    eps = 1e-15
    probs_clipped = np.clip(probs, eps, 1 - eps)

    # BCE formula
    loss = -(targets * np.log(probs_clipped) +
             (1 - targets) * np.log(1 - probs_clipped))

    return np.mean(loss)


def binary_cross_entropy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute binary cross-entropy from logits (more stable).

    Uses the log-sum-exp trick for numerical stability.
    """
    # Numerically stable implementation
    # BCE = log(1 + exp(-z)) when y=1, log(1 + exp(z)) when y=0
    # Can be written as: max(z, 0) - z*y + log(1 + exp(-|z|))

    max_val = np.maximum(logits, 0)
    loss = max_val - logits * targets + np.log(1 + np.exp(-np.abs(logits)))

    return np.mean(loss)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute KL divergence KL(p || q) = sum(p * log(p/q)).

    Measures how different q is from p.
    KL(p || q) >= 0, with equality iff p = q.

    Note: KL divergence is NOT symmetric: KL(p||q) != KL(q||p)
    """
    # Add epsilon to prevent division by zero
    eps = 1e-15
    p_safe = np.clip(p, eps, 1)
    q_safe = np.clip(q, eps, 1)

    # KL(p || q) = sum(p * log(p/q))
    kl = np.sum(p_safe * np.log(p_safe / q_safe))

    return kl


def kl_divergence_from_logprobs(log_p: np.ndarray, log_q: np.ndarray) -> float:
    """
    Compute KL divergence from log-probabilities (more stable).

    KL(p || q) = sum(exp(log_p) * (log_p - log_q))
    """
    p = np.exp(log_p)
    kl = np.sum(p * (log_p - log_q))
    return kl
```

**Helper function - Log-Sum-Exp (for reference):**
```python
def logsumexp(x: np.ndarray, axis: int = None) -> Union[float, np.ndarray]:
    """
    Compute log(sum(exp(x))) in a numerically stable way.

    This is a fundamental building block for many operations.
    """
    x_max = np.max(x, axis=axis, keepdims=True)

    # log(sum(exp(x))) = max + log(sum(exp(x - max)))
    result = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))

    if axis is not None:
        result = result.squeeze(axis)
    else:
        result = result.item()

    return result
```

**Testing:**
```python
import numpy as np

# Test 1: Logits to probabilities
print("=" * 70)
print("TEST 1: Logits to Probabilities")
print("=" * 70)

logits = np.array([[2.0, 1.0, 0.1]])
probs = logits_to_probs(logits)
print(f"Logits: {logits}")
print(f"Probs:  {probs}")
print(f"Sum:    {np.sum(probs, axis=1)}")  # Should be 1.0
print(f"Max prob at index: {np.argmax(probs)}")  # Should be 0 (highest logit)

# Test 2: Numerical stability - very large logits
print("\n" + "=" * 70)
print("TEST 2: Numerical Stability (Large Logits)")
print("=" * 70)

large_logits = np.array([[1000.0, 999.0, 998.0]])
print(f"Large logits: {large_logits}")

# Naive softmax would overflow
# naive = np.exp(large_logits) / np.sum(np.exp(large_logits))  # Would be inf/inf!

# Our stable version works
probs_stable = logits_to_probs(large_logits)
print(f"Stable probs: {probs_stable}")
print(f"Sum: {np.sum(probs_stable)}")  # Still 1.0

# Test 3: Cross-entropy loss
print("\n" + "=" * 70)
print("TEST 3: Cross-Entropy Loss")
print("=" * 70)

logits = np.array([
    [2.0, 1.0, 0.1],   # Predict class 0
    [0.5, 3.0, 0.2],   # Predict class 1
    [0.1, 0.2, 4.0],   # Predict class 2
])
targets = np.array([0, 1, 2])  # All predictions correct

# Method 1: From probabilities (less stable)
probs = logits_to_probs(logits)
loss_from_probs = cross_entropy_loss(probs, targets)
print(f"Loss from probs: {loss_from_probs:.6f}")

# Method 2: From logits (more stable)
loss_from_logits = cross_entropy_from_logits(logits, targets)
print(f"Loss from logits: {loss_from_logits:.6f}")

# Method 3: PyTorch comparison
import torch
import torch.nn.functional as F
logits_torch = torch.tensor(logits, dtype=torch.float32)
targets_torch = torch.tensor(targets, dtype=torch.long)
loss_pytorch = F.cross_entropy(logits_torch, targets_torch).item()
print(f"PyTorch loss: {loss_pytorch:.6f}")
print(f"Match: {np.isclose(loss_from_logits, loss_pytorch)}")

# Test 4: Wrong predictions
print("\n" + "=" * 70)
print("TEST 4: Wrong Predictions (Higher Loss)")
print("=" * 70)

wrong_targets = np.array([2, 0, 1])  # All wrong!
loss_wrong = cross_entropy_from_logits(logits, wrong_targets)
print(f"Loss with correct predictions: {loss_from_logits:.6f}")
print(f"Loss with wrong predictions:  {loss_wrong:.6f}")
print(f"Higher loss when wrong: {loss_wrong > loss_from_logits}")

# Test 5: Binary cross-entropy
print("\n" + "=" * 70)
print("TEST 5: Binary Cross-Entropy")
print("=" * 70)

probs_binary = np.array([0.9, 0.8, 0.3, 0.1])
targets_binary = np.array([1, 1, 0, 0])

bce = binary_cross_entropy(probs_binary, targets_binary)
print(f"Predicted probs: {probs_binary}")
print(f"True labels:     {targets_binary}")
print(f"BCE loss:        {bce:.6f}")

# Compare with PyTorch
probs_torch = torch.tensor(probs_binary, dtype=torch.float32)
targets_torch = torch.tensor(targets_binary, dtype=torch.float32)
bce_pytorch = F.binary_cross_entropy(probs_torch, targets_torch).item()
print(f"PyTorch BCE:     {bce_pytorch:.6f}")
print(f"Match: {np.isclose(bce, bce_pytorch)}")

# Test 6: KL Divergence
print("\n" + "=" * 70)
print("TEST 6: KL Divergence")
print("=" * 70)

p = np.array([0.5, 0.3, 0.2])  # True distribution
q = np.array([0.4, 0.4, 0.2])  # Approximate distribution

kl_pq = kl_divergence(p, q)
kl_qp = kl_divergence(q, p)

print(f"True distribution p: {p}")
print(f"Approx distribution q: {q}")
print(f"KL(p || q): {kl_pq:.6f}")
print(f"KL(q || p): {kl_qp:.6f}")
print(f"KL is not symmetric: {kl_pq != kl_qp}")
print(f"Both are >= 0: {kl_pq >= 0 and kl_qp >= 0}")

# Test 7: Understanding log-space
print("\n" + "=" * 70)
print("TEST 7: Logits → Log-Probs → Probs")
print("=" * 70)

logits_single = np.array([2.0, 1.0, 0.1])

# Method 1: Logits → Probs → Log-Probs (less stable)
probs = logits_to_probs(logits_single)
log_probs_v1 = np.log(probs)

# Method 2: Logits → Log-Probs directly (more stable)
log_probs_v2 = logits_to_logprobs(logits_single)

print(f"Logits:           {logits_single}")
print(f"Probs:            {probs}")
print(f"Log-probs (v1):   {log_probs_v1}")
print(f"Log-probs (v2):   {log_probs_v2}")
print(f"Match: {np.allclose(log_probs_v1, log_probs_v2)}")

# Verify exp(log_probs) = probs
probs_recovered = np.exp(log_probs_v2)
print(f"Recovered probs:  {probs_recovered}")
print(f"Match original:   {np.allclose(probs, probs_recovered)}")
```

**Common mistakes:**
1. ❌ `np.exp(logits) / np.sum(np.exp(logits))` without subtracting max → overflow
2. ❌ `np.log(softmax(logits))` instead of direct log-softmax → numerical errors
3. ❌ Computing cross-entropy from probabilities instead of logits → less stable
4. ❌ Not clipping probabilities before `log()` → log(0) = -inf
5. ❌ Confusing logits, probabilities, and log-probabilities
6. ❌ Forgetting that KL divergence is not symmetric

## Follow-up Questions

**Conceptual:**
- Why is computing cross-entropy from logits more stable than from probabilities?
- What is the relationship between cross-entropy and KL divergence?
- When would you use log-probabilities instead of probabilities?
- Why does the log-sum-exp trick prevent overflow?

**Implementation:**
- How would you extend this to label smoothing?
- Implement focal loss (used in object detection)
- How would you compute gradients of these losses?
- Implement temperature scaling for calibration

**Numerical:**
- What happens if you don't subtract max in softmax?
- Why is `max(z, 0) - z*y + log(1 + exp(-|z|))` numerically stable for BCE?
- How small can probabilities get before numerical issues occur?

**Mathematical:**
- Prove that KL(p || q) >= 0 with equality iff p = q
- What is the relationship between CE, KL, and entropy?
  - CE(p, q) = H(p) + KL(p || q)
- Why is cross-entropy called "cross" entropy?

## Related Concepts
- Softmax and log-softmax (code002)
- Cross-entropy loss (code004)
- Numerical stability techniques
- Information theory (entropy, KL divergence)
- Log-sum-exp trick
- PyTorch's `F.cross_entropy()` implementation

## Key Takeaways

1. **Always compute from logits when possible** - it's more numerically stable
2. **Use log-sum-exp trick** - prevents overflow in exp()
3. **Clip probabilities before log** - prevents log(0)
4. **Understand the relationships**:
   - Logits → Softmax → Probabilities
   - Logits → Log-Softmax → Log-Probabilities
   - Cross-Entropy = Negative Log-Likelihood
   - CE(p, q) = H(p) + KL(p || q)
