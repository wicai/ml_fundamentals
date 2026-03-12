# Implement and Debug Backpropagation in NumPy

**Category:** coding
**Difficulty:** 4
**Tags:** coding, backprop, broadcasting, gradients, numpy, interview-prep
**Source:** OpenAI Research Engineer technical screen (PracHub)

## Question

Implement backpropagation for a two-layer neural network using only NumPy:

**Architecture:** `Input → Affine(W1, b1) → ReLU → Affine(W2, b2) → Softmax → Cross-Entropy Loss`

Your implementation must:
1. Implement the full forward pass with correct shapes at each step
2. Implement numerically stable softmax cross-entropy
3. Derive and implement analytical gradients for W1, b1, W2, b2 via the chain rule — **no autograd, no batch loops**
4. Implement gradient checking via central finite differences to verify correctness
5. Train on synthetic data and show decreasing loss

**Pay close attention to:**
- Shapes and broadcasting at every step (annotate with comments)
- The `b1` gradient requires a reduction (sum over batch) — broadcasting pitfall
- ReLU backward mask must match the forward mask exactly
- Softmax-cross-entropy gradient has an elegant closed form

**Starter code:**
```python
import numpy as np

np.random.seed(42)

class TwoLayerNet:
    def __init__(self, D_in, H, D_out):
        """
        D_in: input dimension (e.g., 784)
        H: hidden dimension (e.g., 128)
        D_out: output classes (e.g., 10)
        """
        # He initialization
        self.W1 = np.random.randn(D_in, H) * np.sqrt(2.0 / D_in)  # (D_in, H)
        self.b1 = np.zeros((1, H))                                  # (1, H)
        self.W2 = np.random.randn(H, D_out) * np.sqrt(2.0 / H)    # (H, D_out)
        self.b2 = np.zeros((1, D_out))                              # (1, D_out)

    def forward(self, X):
        """
        X: (N, D_in)
        Returns: loss (scalar), cache for backward
        """
        pass

    def backward(self, cache):
        """
        Returns: dict of gradients {dW1, db1, dW2, db2}
        """
        pass

    def numerical_gradient(self, X, Y, param_name, eps=1e-5):
        """
        Central finite differences: (f(x+eps) - f(x-eps)) / (2*eps)
        Returns: numerical gradient with same shape as parameter
        """
        pass
```

**Expected completion time:** 45 minutes

## Answer

**Key shape annotations through the network:**
```
Forward:
  X:       (N, D_in)
  z1:      (N, D_in) @ (D_in, H) + (1, H)  → (N, H)     [b1 broadcasts across batch]
  a1:      ReLU(z1) → (N, H)
  z2:      (N, H) @ (H, D_out) + (1, D_out) → (N, D_out) [b2 broadcasts across batch]
  probs:   softmax(z2) → (N, D_out)
  loss:    scalar (mean cross-entropy)

Backward:
  dz2:     probs - one_hot(Y) → (N, D_out)                [softmax-CE gradient]
  dW2:     a1.T @ dz2 → (H, D_out)                        [(H, N) @ (N, D_out)]
  db2:     dz2.sum(axis=0, keepdims=True) → (1, D_out)    [BROADCAST REDUCTION]
  da1:     dz2 @ W2.T → (N, H)
  dz1:     da1 * (z1 > 0) → (N, H)                        [ReLU mask]
  dW1:     X.T @ dz1 → (D_in, H)                          [(D_in, N) @ (N, H)]
  db1:     dz1.sum(axis=0, keepdims=True) → (1, H)        [BROADCAST REDUCTION]
```

**Reference implementation:**
```python
import numpy as np

np.random.seed(42)

class TwoLayerNet:
    def __init__(self, D_in, H, D_out):
        self.W1 = np.random.randn(D_in, H) * np.sqrt(2.0 / D_in)  # (D_in, H)
        self.b1 = np.zeros((1, H))                                  # (1, H)
        self.W2 = np.random.randn(H, D_out) * np.sqrt(2.0 / H)    # (H, D_out)
        self.b2 = np.zeros((1, D_out))                              # (1, D_out)

    def forward(self, X, Y):
        """
        X: (N, D_in)
        Y: (N,) integer class labels
        Returns: loss (scalar), cache
        """
        N = X.shape[0]

        # --- Layer 1: Affine ---
        z1 = X @ self.W1 + self.b1          # (N, D_in) @ (D_in, H) + (1, H) -> (N, H)
                                              # b1 broadcasts (1, H) -> (N, H)

        # --- ReLU ---
        a1 = np.maximum(z1, 0)               # (N, H)

        # --- Layer 2: Affine ---
        z2 = a1 @ self.W2 + self.b2          # (N, H) @ (H, D_out) + (1, D_out) -> (N, D_out)
                                              # b2 broadcasts (1, D_out) -> (N, D_out)

        # --- Numerically stable softmax ---
        z2_stable = z2 - z2.max(axis=1, keepdims=True)  # (N, D_out) - (N, 1) -> (N, D_out)
                                                          # keepdims=True is critical for broadcasting
        exp_z2 = np.exp(z2_stable)                        # (N, D_out)
        probs = exp_z2 / exp_z2.sum(axis=1, keepdims=True)  # (N, D_out) / (N, 1) -> (N, D_out)
                                                              # broadcasting the sum across classes

        # --- Cross-entropy loss ---
        # Select prob of correct class for each sample
        correct_log_probs = -np.log(probs[np.arange(N), Y] + 1e-12)  # (N,)
        loss = correct_log_probs.mean()                                # scalar

        cache = (X, z1, a1, probs, Y, N)
        return loss, cache

    def backward(self, cache):
        """
        Returns: dict of gradients
        """
        X, z1, a1, probs, Y, N = cache

        # --- Softmax-CE gradient (elegant closed form) ---
        # dL/dz2 = probs - one_hot(Y), averaged over batch
        dz2 = probs.copy()                    # (N, D_out)
        dz2[np.arange(N), Y] -= 1             # subtract 1 at correct class
        dz2 /= N                              # average over batch

        # --- Layer 2 gradients ---
        dW2 = a1.T @ dz2                      # (H, N) @ (N, D_out) -> (H, D_out)
        db2 = dz2.sum(axis=0, keepdims=True)  # (N, D_out) -> (1, D_out)
                                               # MUST sum over batch dim — this is the
                                               # reverse of the broadcast in forward pass.
                                               # keepdims=True to match b2 shape (1, D_out)

        # --- Backprop through Layer 2 affine ---
        da1 = dz2 @ self.W2.T                 # (N, D_out) @ (D_out, H) -> (N, H)

        # --- Backprop through ReLU ---
        dz1 = da1 * (z1 > 0).astype(float)   # (N, H) * (N, H) -> (N, H)
                                               # mask from FORWARD z1, not a1

        # --- Layer 1 gradients ---
        dW1 = X.T @ dz1                       # (D_in, N) @ (N, H) -> (D_in, H)
        db1 = dz1.sum(axis=0, keepdims=True)  # (N, H) -> (1, H)
                                               # same broadcast reduction as db2

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    def numerical_gradient(self, X, Y, param_name, eps=1e-5):
        """Central finite differences for gradient checking."""
        param = getattr(self, param_name)
        grad = np.zeros_like(param)

        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old_val = param[idx]

            param[idx] = old_val + eps
            loss_plus, _ = self.forward(X, Y)

            param[idx] = old_val - eps
            loss_minus, _ = self.forward(X, Y)

            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            param[idx] = old_val
            it.iternext()

        return grad


# ===================== TESTING =====================

# Synthetic data
N, D_in, H, D_out = 100, 20, 50, 5
X = np.random.randn(N, D_in)
Y = np.random.randint(0, D_out, (N,))

net = TwoLayerNet(D_in, H, D_out)

# --- Gradient check ---
print("=== Gradient Check ===")
loss, cache = net.forward(X, Y)
grads = net.backward(cache)

for param_name in ['W1', 'b1', 'W2', 'b2']:
    num_grad = net.numerical_gradient(X, Y, param_name)
    ana_grad = grads['d' + param_name]
    rel_error = np.abs(num_grad - ana_grad).max() / (np.abs(num_grad).max() + 1e-8)
    print(f"  {param_name}: max relative error = {rel_error:.2e} {'PASS' if rel_error < 1e-5 else 'FAIL'}")

# --- Training ---
print("\n=== Training ===")
lr = 0.1
for epoch in range(100):
    loss, cache = net.forward(X, Y)
    grads = net.backward(cache)

    net.W1 -= lr * grads['dW1']
    net.b1 -= lr * grads['db1']
    net.W2 -= lr * grads['dW2']
    net.b2 -= lr * grads['db2']

    if epoch % 10 == 0:
        probs = cache[3]  # from forward
        preds = probs.argmax(axis=1)
        acc = (preds == Y).mean()
        print(f"  Epoch {epoch:3d}: loss={loss:.4f}, acc={acc:.4f}")
```

**Common mistakes:**
1. ❌ **Forgetting `keepdims=True`** in softmax normalization — `z2.max(axis=1)` gives `(N,)`, which broadcasts incorrectly against `(N, D_out)` in NumPy (it works but subtracts wrong values)
2. ❌ **Bias gradient without sum** — writing `db2 = dz2` instead of `dz2.sum(axis=0)`. The forward pass broadcasts `(1, D_out)` to `(N, D_out)`, so backward must sum over the batch dimension. This is the fundamental rule: **gradient of a broadcast is a reduction (sum)**
3. ❌ **Missing `keepdims=True` in bias gradient** — `dz2.sum(axis=0)` gives shape `(D_out,)` instead of `(1, D_out)`. Works in NumPy due to broadcasting but conceptually wrong and can cause bugs in more complex networks
4. ❌ **ReLU mask from `a1` instead of `z1`** — `(a1 > 0)` and `(z1 > 0)` differ when `z1 == 0` exactly. Use `z1 > 0` to match the mathematical derivative
5. ❌ **Forgetting `/N` in the softmax-CE gradient** — the loss is averaged over batch, so the gradient must be too. Missing this makes gradients N times too large
6. ❌ **Using `z2` instead of `z2_stable` in softmax** — overflow for large logits

## Key Broadcasting Rules Tested

| Operation | Forward Shape | Backward Shape | Rule |
|-----------|--------------|----------------|------|
| `X @ W1 + b1` | `(N,H) + (1,H)` | `db1 = sum over axis=0` | **Broadcast ↔ Reduce** |
| `z2 - z2.max(axis=1, keepdims=True)` | `(N,D) - (N,1)` | — | Must use `keepdims` |
| `exp / exp.sum(axis=1, keepdims=True)` | `(N,D) / (N,1)` | — | Must use `keepdims` |
| `a1 @ W2 + b2` | `(N,D) + (1,D)` | `db2 = sum over axis=0` | **Broadcast ↔ Reduce** |

**The core insight:** Every broadcast in the forward pass becomes a `sum` reduction in the backward pass. If you add `b` with shape `(1, H)` to `z` with shape `(N, H)`, then `db = dz.sum(axis=0, keepdims=True)`.

## Follow-up Questions
- Why is the softmax-cross-entropy gradient simply `probs - one_hot(Y)`? Derive it.
- What happens if you initialize all weights to zero? Why?
- How would you extend this to support batch normalization? What are the backward pass shapes?
- How would you add L2 regularization? Where does the gradient change?
