# "Stupid Bugs" ML Debugging: Broadcasting Edition

**Category:** coding
**Difficulty:** 3
**Tags:** coding, debugging, broadcasting, tensor-shapes, interview-prep
**Source:** OpenAI/DeepMind-style ML debugging round (Sundeep Teki guide)

## Question

The following Jupyter notebook trains a simple 2-layer MLP on MNIST-style data. It runs without any errors or warnings, but the model **doesn't learn** — accuracy stays at ~10% (random chance for 10 classes).

There are **7 bugs**. All of them involve silent broadcasting, wrong dimensions, or shape mismatches that PyTorch/NumPy happily execute without complaint. Find and fix all of them.

**Buggy code:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

# --- Synthetic MNIST-like data ---
N = 5000
X = torch.randn(N, 784)
Y = torch.randint(0, 10, (N,))
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=64, shuffle=False)           # BUG 1

# --- Model ---
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(784, 256))
        self.b1 = nn.Parameter(torch.zeros(256))
        self.w2 = nn.Parameter(torch.randn(256, 10))
        self.b2 = nn.Parameter(torch.zeros(10))
        self.bn_mean = torch.zeros(256)                              # BUG 2
        self.bn_var = torch.ones(256)

    def forward(self, x):
        # Layer 1
        h = x @ self.w1 + self.b1                                   # (batch, 256)

        # "Batch norm" (manual)
        mean = h.mean(dim=0, keepdim=True)                           # (1, 256)
        var = h.var(dim=0, keepdim=True)                             # (1, 256)
        h = (h - self.bn_mean) / (self.bn_var + 1e-5).sqrt()        # BUG 3

        # ReLU
        h = h * (h > 0).float()

        # Layer 2
        logits = h @ self.w2 + self.b2                               # (batch, 10)

        # Per-sample loss weighting based on confidence
        confidence = logits.softmax(dim=-1).max(dim=-1).values       # (batch,)
        weight = 1.0 - confidence                                    # (batch,)
        logits = logits * weight                                     # BUG 4

        return logits

model = MLP()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# --- Training ---
for epoch in range(10):
    total_loss = 0
    correct = 0
    total = 0
    for xb, yb in loader:
        logits = model(xb)

        # Label smoothing
        n_classes = 10
        smooth = 0.1
        one_hot = torch.zeros(yb.size(0), n_classes)
        one_hot[torch.arange(yb.size(0)), yb] = 1.0
        smooth_labels = one_hot * (1 - smooth) + smooth / n_classes  # (batch, 10)

        # Cross-entropy with smooth labels
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_labels * log_probs).sum(dim=-1).mean()

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        max_norm = 1.0
        total_norm = torch.norm(
            torch.stack([p.grad.norm() for p in model.parameters()])  # BUG 5
        )
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in model.parameters():
                p.grad.data.mul_(clip_coef)

        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    # Learning rate schedule: cosine decay
    for pg in optimizer.param_groups:
        pg['lr'] = 0.01 * (1 + torch.cos(torch.tensor(epoch / 10 * 3.14159))) / 2  # BUG 6

    acc = correct / total
    print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}, acc={acc:.4f}")

# --- Evaluation ---
with torch.no_grad():
    all_logits = model(X)
    preds = all_logits.argmax(dim=-1)
    final_acc = (preds == Y).float().mean()
    print(f"Final accuracy: {final_acc:.4f}")
```

## Answer

### Bug 1: DataLoader not shuffling
- **Symptom:** Model sees data in same order every epoch — learns spurious sequential patterns
- **Root cause:** `shuffle=False`
- **Fix:** `shuffle=True`
- **Why it's subtle:** Code runs fine, loss decreases slightly, but accuracy never gets good

### Bug 2: Running stats are plain tensors, not Parameters or buffers
- **Symptom:** `self.bn_mean` and `self.bn_var` are not on the same device as inputs and are never updated
- **Root cause:** `torch.zeros(256)` creates a regular tensor, not tracked by the module
- **Fix:** Use `self.register_buffer('bn_mean', torch.zeros(256))` or just use the batch statistics (see Bug 3)

### Bug 3: Batch norm uses stale stats instead of batch stats (broadcasting silently works)
- **Symptom:** `h - self.bn_mean` broadcasts `(batch, 256) - (256,)` which "works" shape-wise. But `self.bn_mean` is always zeros and `self.bn_var` is always ones, so normalization is `(h - 0) / 1` = no-op. Meanwhile `mean` and `var` are computed but **never used**.
- **Root cause:** Should use `mean` and `var` from the current batch, not the stale `self.bn_mean`/`self.bn_var`
- **Fix:** `h = (h - mean) / (var + 1e-5).sqrt()` — use the batch statistics that were just computed
- **Broadcasting lesson:** `(256,)` broadcasts against `(batch, 256)` without error, masking the bug

### Bug 4: Confidence weighting broadcasts wrong — scales logits by a per-sample scalar
- **Symptom:** `logits * weight` where logits is `(batch, 10)` and weight is `(batch,)`. This **fails** without unsqueeze — but if it did work, it would zero out logits for confident predictions, destroying the loss signal. The actual bug is conceptual: this "confidence weighting" kills gradients for well-classified samples and should be removed or applied to the loss, not the logits.
- **Root cause:** `weight` is `(batch,)` and needs `unsqueeze(-1)` to broadcast against `(batch, 10)`. But more fundamentally, scaling logits by `(1 - confidence)` makes training unstable — high-confidence correct predictions get zero gradient.
- **Fix:** Remove this block entirely. If you want confidence-based weighting, apply it to the loss, not to logits:
  ```python
  # Just return logits without weighting
  return logits
  ```
- **Broadcasting lesson:** If weight had shape `(batch, 1)`, it would silently broadcast against `(batch, 10)` and the training would be broken in a hard-to-debug way

### Bug 5: Gradient clipping detaches from computation graph
- **Symptom:** `torch.stack([p.grad.norm() for p in model.parameters()])` — this computes norms fine, but building a list comprehension then stacking is fragile. The real issue: if any param has `grad=None` (e.g., unused params), this crashes.
- **Root cause:** Should use `torch.nn.utils.clip_grad_norm_` which handles edge cases
- **Fix:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

### Bug 6: Learning rate becomes a tensor, not a float
- **Symptom:** `torch.cos(...)` returns a tensor. Setting `pg['lr']` to a tensor instead of a float can cause subtle issues in some PyTorch versions — the optimizer may not step correctly
- **Root cause:** `torch.cos(...)` returns a 0-dim tensor
- **Fix:** Add `.item()`: `pg['lr'] = (0.01 * (1 + torch.cos(torch.tensor(epoch / 10 * 3.14159))) / 2).item()`

### Bug 7 (Bonus): Weight initialization is wrong
- **Symptom:** `torch.randn(784, 256)` has std=1.0, which is way too large for a 784-input layer. Activations explode or saturate.
- **Root cause:** Should use `torch.randn(784, 256) * (2.0 / 784) ** 0.5` (He init) or similar
- **Fix:** Scale initialization: `self.w1 = nn.Parameter(torch.randn(784, 256) * (2.0/784)**0.5)` and `self.w2 = nn.Parameter(torch.randn(256, 10) * (2.0/256)**0.5)`

### Key Broadcasting Takeaways

The "silent broadcasting" bugs are the most dangerous:
- **Bug 3** is the worst: `(256,)` broadcasts perfectly against `(batch, 256)`, producing a valid result that is completely wrong
- **Bug 4**: `(batch,) * (batch, 10)` — shape mismatch that either errors or silently broadcasts if reshaped
- These bugs never raise errors, shapes always "work" — you can only catch them by reasoning about what the values *should be*

## Follow-up Questions
- How would you write a unit test to catch Bug 3 (stale batch norm stats)?
- What's the difference between `view` and `reshape`? When does `view` fail?
- If Bug 4's weight had shape `(batch, 1)` instead of `(batch,)`, what would happen and why?
