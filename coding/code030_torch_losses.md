# Use PyTorch Loss Functions Correctly

**Category:** coding
**Difficulty:** 2
**Tags:** coding, pytorch, loss functions, training

## Question
Demonstrate correct usage of PyTorch's built-in loss functions. This is about knowing the right API calls, input formats, and common gotchas — not implementing from scratch.

Write a function for each scenario that creates the right loss function and computes the loss. Pay attention to:
- Whether inputs should be logits or probabilities
- The correct tensor shapes and dtypes
- When to use `_with_logits` variants

**Function signature:**
```python
def multiclass_classification_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute loss for multiclass classification (e.g., image classification).

    Args:
        logits: raw model outputs, shape (batch_size, num_classes) — NOT softmaxed
        targets: class indices, shape (batch_size,), dtype long
    Returns:
        loss: scalar tensor
    """
    pass

def binary_classification_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute loss for binary classification (e.g., spam detection).

    Args:
        logits: raw model outputs, shape (batch_size,) — NOT sigmoidified
        targets: binary labels, shape (batch_size,), values in {0, 1}, dtype float
    Returns:
        loss: scalar tensor
    """
    pass

def regression_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE loss for regression (e.g., predicting price).

    Args:
        predictions: model outputs, shape (batch_size,)
        targets: true values, shape (batch_size,)
    Returns:
        loss: scalar tensor
    """
    pass

def multilabel_classification_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute loss for multi-label classification (e.g., image tags — each image
    can have multiple labels simultaneously).

    Args:
        logits: raw model outputs, shape (batch_size, num_labels)
        targets: binary indicators, shape (batch_size, num_labels), dtype float
    Returns:
        loss: scalar tensor
    """
    pass

def sequence_classification_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Compute loss for token-level classification (e.g., language modeling),
    ignoring padding tokens.

    Args:
        logits: shape (batch_size, seq_len, vocab_size)
        targets: shape (batch_size, seq_len), with ignore_index for padding
        ignore_index: index to ignore in loss computation
    Returns:
        loss: scalar tensor
    """
    pass
```

## Answer

**Key concepts:**
1. `F.cross_entropy` expects **logits** (raw scores), NOT softmax outputs
2. `F.binary_cross_entropy_with_logits` is more stable than `F.binary_cross_entropy`
3. Shape matters: cross_entropy wants `(N, C)` logits, `(N,)` targets
4. For sequence models, reshape to `(N*T, C)` and `(N*T,)`
5. `ignore_index` skips padding tokens in loss computation

**Reference implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def multiclass_classification_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Use F.cross_entropy — it applies log-softmax + NLL internally.
    Do NOT apply softmax before passing to this function!

    Input: logits (N, C), targets (N,) as class indices (long)
    """
    return F.cross_entropy(logits, targets)

def binary_classification_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Use binary_cross_entropy_with_logits — it applies sigmoid internally.
    More numerically stable than sigmoid + binary_cross_entropy.

    Input: logits (N,), targets (N,) as float {0.0, 1.0}
    """
    return F.binary_cross_entropy_with_logits(logits, targets)

def regression_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Use F.mse_loss for mean squared error.
    Could also use F.l1_loss (MAE) or F.smooth_l1_loss (Huber).

    Input: predictions (N,), targets (N,)
    """
    return F.mse_loss(predictions, targets)

def multilabel_classification_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Multi-label = independent binary classification per label.
    Use binary_cross_entropy_with_logits on the full (N, L) tensor.

    Input: logits (N, L), targets (N, L) as float
    """
    return F.binary_cross_entropy_with_logits(logits, targets)

def sequence_classification_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    For token-level prediction: reshape (N, T, C) → (N*T, C).
    Use ignore_index to skip padding tokens.

    Input: logits (N, T, C), targets (N, T) with padding as ignore_index
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Reshape: (N, T, C) → (N*T, C) and (N, T) → (N*T,)
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    return F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)
```

**Testing:**
```python
import torch
import torch.nn.functional as F

torch.manual_seed(1)

# Test 1: Multiclass classification
print("=" * 70)
print("TEST 1: Multiclass Classification (F.cross_entropy)")
print("=" * 70)
logits = torch.randn(4, 5)  # batch=4, classes=5
targets = torch.tensor([0, 3, 1, 4])  # class indices
loss = multiclass_classification_loss(logits, targets)
print(f"Logits shape: {logits.shape}")
print(f"Targets: {targets}")
print(f"Loss: {loss.item():.4f}")

# Verify: applying softmax first is WRONG
wrong_loss = F.cross_entropy(torch.softmax(logits, dim=1), targets)
print(f"WRONG (softmax then CE): {wrong_loss.item():.4f}")
print(f"These differ because CE applies log-softmax internally!")

# Test 2: Binary classification
print("\n" + "=" * 70)
print("TEST 2: Binary Classification (BCE with logits)")
print("=" * 70)
logits = torch.tensor([2.0, -1.0, 0.5, -3.0])
targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
loss = binary_classification_loss(logits, targets)
print(f"Logits: {logits}")
print(f"Targets: {targets}")
print(f"Loss: {loss.item():.4f}")

# Verify stability: BCE with logits handles extreme values
extreme_logits = torch.tensor([100.0, -100.0])
extreme_targets = torch.tensor([1.0, 0.0])
loss_stable = F.binary_cross_entropy_with_logits(extreme_logits, extreme_targets)
print(f"Extreme logits loss (stable): {loss_stable.item():.4f}")
# Naive: sigmoid + BCE would fail here
# sigmoid_out = torch.sigmoid(extreme_logits)  # [1.0, 0.0] exactly
# F.binary_cross_entropy(sigmoid_out, extreme_targets)  # log(0) = -inf!

# Test 3: Regression
print("\n" + "=" * 70)
print("TEST 3: Regression (MSE)")
print("=" * 70)
predictions = torch.tensor([2.5, 0.0, 3.0])
targets = torch.tensor([3.0, -0.5, 2.8])
loss = regression_loss(predictions, targets)
expected = ((predictions - targets) ** 2).mean()
print(f"Predictions: {predictions}")
print(f"Targets: {targets}")
print(f"MSE Loss: {loss.item():.4f}")
print(f"Manual MSE: {expected.item():.4f}")
print(f"Match: {torch.isclose(loss, expected)}")

# Test 4: Multi-label classification
print("\n" + "=" * 70)
print("TEST 4: Multi-Label Classification")
print("=" * 70)
logits = torch.randn(3, 4)  # batch=3, labels=4
targets = torch.tensor([
    [1, 0, 1, 0],  # labels 0 and 2 active
    [0, 1, 0, 1],  # labels 1 and 3 active
    [1, 1, 0, 0],  # labels 0 and 1 active
], dtype=torch.float)
loss = multilabel_classification_loss(logits, targets)
print(f"Logits shape: {logits.shape}")
print(f"Targets:\n{targets}")
print(f"Loss: {loss.item():.4f}")

# Test 5: Sequence classification with padding
print("\n" + "=" * 70)
print("TEST 5: Sequence Classification (Language Model)")
print("=" * 70)
batch_size, seq_len, vocab_size = 2, 5, 100
logits = torch.randn(batch_size, seq_len, vocab_size)
# Targets with padding (-100 means ignore)
targets = torch.tensor([
    [12, 45, 67, -100, -100],  # 3 real tokens, 2 padding
    [23, 56, 89, 11,  -100],   # 4 real tokens, 1 padding
])
loss = sequence_classification_loss(logits, targets)
print(f"Logits shape: {logits.shape}")
print(f"Targets shape: {targets.shape}")
print(f"Loss: {loss.item():.4f}")

# Verify that padding is ignored
targets_no_pad = torch.tensor([
    [12, 45, 67, 0, 0],
    [23, 56, 89, 11, 0],
])
loss_no_ignore = sequence_classification_loss(logits, targets_no_pad, ignore_index=-999)
print(f"Loss without ignore: {loss_no_ignore.item():.4f}")
print(f"Losses differ (padding matters): {not torch.isclose(loss, loss_no_ignore)}")

# Test 6: Reduction modes
print("\n" + "=" * 70)
print("TEST 6: Reduction Modes")
print("=" * 70)
logits = torch.randn(4, 3)
targets = torch.tensor([0, 1, 2, 0])
print(f"reduction='mean':  {F.cross_entropy(logits, targets, reduction='mean').item():.4f}")
print(f"reduction='sum':   {F.cross_entropy(logits, targets, reduction='sum').item():.4f}")
print(f"reduction='none':  {F.cross_entropy(logits, targets, reduction='none').tolist()}")
```

**Common mistakes:**
1. Applying softmax before `F.cross_entropy` (it does log-softmax internally!)
2. Applying sigmoid before `F.binary_cross_entropy_with_logits` (double sigmoid!)
3. Using `F.binary_cross_entropy` with logits (wrong results AND numerically unstable)
4. Wrong target dtype: `F.cross_entropy` wants `long`, BCE wants `float`
5. Forgetting to reshape `(N, T, C)` → `(N*T, C)` for sequence models
6. Not using `ignore_index` for padding tokens

## Follow-up Questions
- Why does `F.cross_entropy` take logits instead of probabilities?
- When would you use `reduction='none'` vs `'mean'`?
- What's the difference between `nn.CrossEntropyLoss()` and `F.cross_entropy()`?
- How would you add label smoothing to cross-entropy?
- When would you use `F.smooth_l1_loss` instead of MSE?
