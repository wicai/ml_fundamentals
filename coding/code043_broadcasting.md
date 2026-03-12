# PyTorch Broadcasting

**Category:** coding
**Difficulty:** 3
**Tags:** coding, broadcasting, tensors, shapes, numpy

## Question

Broadcasting lets PyTorch operate on tensors of different shapes without copying data. Mastering it is essential for writing clean, vectorized ML code — and it's a frequent topic in Anthropic/OpenAI research engineer interviews (via the "Tensor Puzzles" style, or as part of implementing attention, masking, and loss functions).

### Part 1: Shape prediction

For each expression below, state the output shape or explain why it raises an error. No code — reason from the rules.

```python
import torch

a = torch.ones(5, 1, 4, 1)
b = torch.ones(   3, 1, 6)

# Q1: What is (a + b).shape?

# Q2: What is (a * b).shape?

c = torch.ones(5, 2, 4, 1)
d = torch.ones(   3, 1, 6)

# Q3: What is (c + d).shape, or does it error? Why?

e = torch.ones(4,)
f = torch.ones(4, 1)

# Q4: What is (e + f).shape? Most people get this wrong.
```

### Part 2: Implement without loops

Implement each function using only PyTorch tensor operations — **no Python loops or list comprehensions**.

```python
import torch

def causal_mask(T: int) -> torch.Tensor:
    """
    Return a boolean mask of shape (T, T) where mask[i, j] is True
    if position j is allowed to attend to position i (i.e. j <= i).
    Used in decoder-only transformers to prevent attending to future tokens.
    """
    pass


def sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Given a 1-D tensor of sequence lengths, return a boolean mask of
    shape (B, max_len) where mask[b, t] is True if t < lengths[b].

    Args:
        lengths: shape (B,) — each value is in [0, max_len]
        max_len: int
    Returns:
        mask: shape (B, max_len), dtype bool
    """
    pass


def pairwise_l2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute the pairwise L2 distance between rows of A and B.

    Args:
        A: shape (m, d)
        B: shape (n, d)
    Returns:
        dists: shape (m, n), where dists[i, j] = ||A[i] - B[j]||_2
    """
    pass


def apply_padding_mask(
    logits: torch.Tensor,
    pad_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Zero out (set to -inf) logit positions corresponding to padding tokens
    before softmax.

    Args:
        logits:   shape (B, T, V) — raw logits over vocabulary
        pad_mask: shape (B, T)    — True where token is real, False where padded
    Returns:
        masked_logits: shape (B, T, V)
    """
    pass
```

### Part 3: Gradient shape prediction

```python
import torch

a = torch.ones(3, 1, requires_grad=True)
b = torch.ones(1, 4)
c = (a + b).sum()
c.backward()

# Q5: What is a.grad.shape, and why?
# Q6: What is the value of every element in a.grad, and why?
```

## Answer

### Part 1: Shape prediction rules

**The two rules:**
1. Align shapes **right-to-left**; pad missing leading dims with 1
2. Each dim pair must be: equal, or one of them is 1 (it stretches)

**Q1 & Q2: `(5, 1, 4, 1) + (3, 1, 6)` → `(5, 3, 4, 6)`**
```
a: (5, 1, 4, 1)
b: (1, 3, 1, 6)  ← left-pad b with 1

dim -1:  1 vs 6  → 6   (a stretches)
dim -2:  4 vs 1  → 4   (b stretches)
dim -3:  1 vs 3  → 3   (a stretches)
dim -4:  5 vs 1  → 5   (b stretches)

result:  (5, 3, 4, 6)
```

**Q3: `(5, 2, 4, 1) + (3, 1, 6)` → RuntimeError**
```
c: (5, 2, 4, 1)
d: (1, 3, 1, 6)

dim -3:  2 vs 3  → ERROR — neither is 1
```

**Q4: `(4,) + (4, 1)` → `(4, 4)` — the common gotcha**
```
e: (   4)  → left-pad → (1, 4)
f: (4, 1)

dim -1:  4 vs 1  → 4
dim -2:  1 vs 4  → 4

result:  (4, 4)  ← NOT (4,) or (4, 1)!
```
Most people expect `(4,)` or `(4, 1)` because the shapes "seem compatible." They're not — broadcasting expands both, producing a (4, 4) outer product.

---

### Part 2: Implementations

```python
import torch


def causal_mask(T: int) -> torch.Tensor:
    i = torch.arange(T)[:, None]  # (T, 1) — query positions
    j = torch.arange(T)[None, :]  # (1, T) — key positions
    return j <= i                  # (T, T) — True where attending is allowed


def sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    # lengths: (B,)  →  (B, 1)  broadcasts against  (1, max_len)
    positions = torch.arange(max_len, device=lengths.device)[None, :]  # (1, max_len)
    return positions < lengths[:, None]                                  # (B, max_len)


def pairwise_l2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A[:, None]: (m, 1, d)   B[None, :]: (1, n, d)
    # difference: (m, n, d) by broadcasting, then reduce over d
    return ((A[:, None] - B[None, :]) ** 2).sum(-1).sqrt()  # (m, n)


def apply_padding_mask(
    logits: torch.Tensor,
    pad_mask: torch.Tensor,
) -> torch.Tensor:
    # pad_mask is (B, T); logits is (B, T, V)
    # unsqueeze to (B, T, 1) so it broadcasts over the V dimension
    return logits.masked_fill(~pad_mask[:, :, None], float('-inf'))
```

---

### Part 3: Gradient shapes

**Q5: `a.grad.shape` is `(3, 1)`** — same shape as `a`.

PyTorch always accumulates gradients into a tensor of the **same shape as the parameter**. When a `(3, 1)` tensor is broadcast to `(3, 4)` during the forward pass, the backward pass sums gradients over the expanded dim (dim=1) and restores shape `(3, 1)`.

**Q6: Every element of `a.grad` is `4.0`**

```
c = (a + b).sum()
  = sum over (3, 4) tensor

dc/da[i, 0] = sum of dc/d(a+b)[i, j] for j in 0..3
            = 1 + 1 + 1 + 1 = 4.0
```

Because `a[i, 0]` contributes to all 4 columns (it broadcast across dim=1), its gradient is the sum of the upstream gradient across those 4 positions. Since `d(sum)/d(anything) = 1`, each position contributes 1, giving grad = 4.

---

## Common Mistakes

1. **Q4 gotcha**: `(4,) + (4, 1)` → `(4, 4)`, not `(4, 1)`. The 1D tensor is left-padded to `(1, 4)`, producing an outer product.
2. **In-place ops can't expand**: `x = torch.zeros(1, 3, 1); x.add_(torch.ones(3, 1, 7))` → RuntimeError. In-place ops cannot change a tensor's shape.
3. **Forgetting `None` indexing**: to control which dim broadcasts, use `a[:, None]` (adds dim on right) vs `a[None, :]` (adds dim on left).
4. **Gradient accumulation**: forgetting that broadcasting in the forward pass means gradient *summation* in the backward pass — the grad shape always matches the original parameter shape.

## Follow-up Questions

- Why does `(4,) + (4, 1)` produce `(4, 4)` and not an error?
- In `pairwise_l2`, could you use the identity `||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b` instead? What are the numerical trade-offs?
- Why can't in-place operations expand a tensor's shape?
- If `a` has shape `(3, 1)` and broadcasts to `(3, 4)`, what shape is `a.grad` after `.backward()`, and what value does each element hold?
- How does sequence masking with `-inf` before softmax produce exactly zero attention weight on padded positions?
