# Debug and Fix a PyTorch Transformer Training Loop

**Category:** coding
**Difficulty:** 4
**Tags:** coding, debugging, broadcasting, attention, transformer, interview-prep
**Source:** OpenAI Research Engineer interview (PracHub)

## Question

The following code implements a minimal causal decoder-only language model. It runs without errors but the loss **never decreases** and sometimes becomes **NaN**. Find and fix all the bugs.

There are **at least 12 distinct defects**. For each bug, identify:
- The symptom (what goes wrong)
- The root cause
- The minimal fix

Then provide corrected, runnable code that shows decreasing loss.

**Buggy code:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

class TinyDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(d_model))          # BUG 1
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)
        self.drop = nn.Dropout(0.2)

    def forward(self, x, attn_mask=None):
        B, S = x.shape
        h = self.tok(x) + self.pos                             # BUG 1 (consumed here)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        H = self.n_heads
        q = q.view(B, S, H, -1)                                # BUG 2
        k = k.view(B, S, H, -1)
        v = v.view(B, S, H, -1)
        attn = torch.matmul(q, k.transpose(-2, -1))            # BUG 3
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9)      # BUG 4
        w = F.softmax(attn, dim=0)                              # BUG 5
        z = torch.matmul(w, v).view(B, S, -1)                  # BUG 6
        h2 = self.proj(z)
        h3 = self.ln(h + self.drop(h2))
        return self.out(h3).softmax(-1)                         # BUG 7

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TinyDecoder(vocab_size=100).to('cpu')               # BUG 8
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.eval()                                                # BUG 9
    for step in range(200):
        x = torch.randint(1, 100, (32, 16), device=device)
        y = x                                                   # BUG 10
        logits = model(x)
        loss = F.cross_entropy(F.log_softmax(logits, 2), y.float())  # BUG 11, 12
        with torch.no_grad():                                   # BUG 13
            loss.backward()
        opt.step()                                              # BUG 14
        if step % 50 == 0:
            print(step, loss.item())

train()
```

## Answer

### Bug List

**Bug 1: Positional embedding is a single vector, not per-position**
- Symptom: Same positional info broadcast to every timestep — model has no sense of position
- Root cause: `self.pos = nn.Parameter(torch.zeros(d_model))` is shape `(d_model,)`, broadcasts identically across all positions
- Fix: `self.pos = nn.Parameter(torch.zeros(max_seq_len, d_model))` and index with `self.pos[:S]`

**Bug 2: Missing head-dimension transpose for attention**
- Symptom: `q.view(B, S, H, d_k)` puts heads in dim 2, but `matmul` operates on last two dims — so attention is computed across heads instead of across sequence positions
- Root cause: Need `(B, H, S, d_k)` for correct `matmul` semantics
- Fix: `.view(B, S, H, -1).transpose(1, 2)` for q, k, v

**Bug 3: Missing scaling factor in attention**
- Symptom: Attention logits have variance proportional to `d_k`, causing softmax to saturate — vanishing gradients
- Root cause: `torch.matmul(q, k.T)` without dividing by `sqrt(d_k)`
- Fix: `attn = attn / (q.size(-1) ** 0.5)`

**Bug 4: No causal mask is ever created or passed**
- Symptom: Model can attend to future tokens — learns to cheat, doesn't generalize
- Root cause: `attn_mask` is never constructed; default is `None` so the `if` block never runs
- Fix: Create `torch.tril(torch.ones(S, S)).bool()` and pass it. Also: the mask should be boolean for `masked_fill`

**Bug 5: Softmax along wrong dimension**
- Symptom: `dim=0` normalizes across the batch — each token's attention weights sum to 1 across different samples, not across key positions
- Root cause: Should be `dim=-1` to normalize over key positions
- Fix: `F.softmax(attn, dim=-1)`

**Bug 6: Reshaping after attention with wrong head ordering**
- Symptom: After fixing Bug 2 to `(B, H, S, d_k)`, need to transpose back before `.view`
- Root cause: `.view(B, S, -1)` on `(B, H, S, d_k)` interleaves head outputs incorrectly
- Fix: `.transpose(1, 2).contiguous().view(B, S, -1)`

**Bug 7: Applying softmax before returning logits**
- Symptom: `F.cross_entropy` expects raw logits, not probabilities. Applying softmax then log_softmax double-squashes — loss plateaus and gradients vanish
- Root cause: `return self.out(h3).softmax(-1)` — should return raw logits
- Fix: `return self.out(h3)`

**Bug 8: Model placed on 'cpu' regardless of device variable**
- Symptom: If CUDA is available, data is on GPU but model is on CPU — runtime error
- Root cause: Hardcoded `.to('cpu')` instead of `.to(device)`
- Fix: `.to(device)`

**Bug 9: Model in eval mode during training**
- Symptom: Dropout is disabled, batch norm (if present) uses running stats — model doesn't regularize
- Root cause: `model.eval()` should be `model.train()`
- Fix: `model.train()`

**Bug 10: Labels are the same as inputs (not shifted)**
- Symptom: For next-token prediction, label at position `t` should be input at position `t+1`
- Root cause: `y = x` means model is trained to predict the current token, not the next one
- Fix: `y = x[:, 1:]` and `logits = logits[:, :-1]` (shift alignment)

**Bug 11: Double log-softmax**
- Symptom: `F.cross_entropy` already applies log_softmax internally. Wrapping with `F.log_softmax` applies it twice — loss can go negative or behave erratically
- Root cause: `F.cross_entropy(F.log_softmax(logits, 2), ...)`
- Fix: `F.cross_entropy(logits, y)` — pass raw logits directly

**Bug 12: Targets should be long integers, not float**
- Symptom: `F.cross_entropy` with class indices expects `torch.long` targets. Passing `y.float()` causes an error or incorrect behavior
- Root cause: `y.float()` converts integer class indices to float
- Fix: Just pass `y` (already `torch.long` from `randint`)

**Bug 13: `loss.backward()` inside `torch.no_grad()`**
- Symptom: Gradients are not computed — optimizer steps are no-ops
- Root cause: `with torch.no_grad()` disables gradient tracking; backward pass produces no gradients
- Fix: Remove the `with torch.no_grad()` wrapper

**Bug 14: Missing `opt.zero_grad()`**
- Symptom: Gradients accumulate across steps, causing unstable/divergent training
- Root cause: No `opt.zero_grad()` before backward
- Fix: Add `opt.zero_grad()` before `loss.backward()`

### Corrected Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

class TinyDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, max_seq_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(max_seq_len, d_model))  # Fix 1
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        B, S = x.shape
        d_k = self.d_model // self.n_heads
        H = self.n_heads

        h = self.tok(x) + self.pos[:S]                              # Fix 1

        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, H, d_k).transpose(1, 2)                   # Fix 2: (B, H, S, d_k)
        k = k.view(B, S, H, d_k).transpose(1, 2)
        v = v.view(B, S, H, d_k).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5) # Fix 3

        # Fix 4: Create causal mask
        causal_mask = torch.tril(torch.ones(S, S, device=x.device)).bool()
        attn = attn.masked_fill(~causal_mask, float('-inf'))

        w = F.softmax(attn, dim=-1)                                 # Fix 5
        w = self.drop(w)

        z = torch.matmul(w, v)                                      # (B, H, S, d_k)
        z = z.transpose(1, 2).contiguous().view(B, S, -1)           # Fix 6

        h2 = self.proj(z)
        h3 = self.ln(h + self.drop(h2))
        return self.out(h3)                                          # Fix 7: raw logits

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TinyDecoder(vocab_size=100).to(device)                   # Fix 8
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()                                                    # Fix 9

    for step in range(200):
        x = torch.randint(1, 100, (32, 16), device=device)
        y = x[:, 1:]                                                # Fix 10: shifted labels
        logits = model(x)[:, :-1]                                   # Fix 10: align logits

        loss = F.cross_entropy(logits.reshape(-1, 100), y.reshape(-1))  # Fix 11, 12

        opt.zero_grad()                                              # Fix 14
        loss.backward()                                              # Fix 13: no torch.no_grad()
        opt.step()

        if step % 50 == 0:
            print(step, loss.item())

train()
```

### Broadcasting Bugs Summary
The most insidious broadcasting bugs in this problem:
- **Bug 1**: `pos` shape `(d_model,)` silently broadcasts to `(B, S, d_model)` — same embedding at every position
- **Bug 2**: Without transpose, `(B, S, H, d_k)` matmul computes attention across the wrong dimensions
- **Bug 5**: `softmax(dim=0)` silently normalizes across batch — shapes are valid, no error thrown

## Follow-up Questions
- Write 3 unit tests that would fail on the buggy code and pass on the fixed code
- What is the time and memory complexity of scaled dot-product attention? Name two strategies to reduce memory usage and their trade-offs.
- How would you add flash attention or gradient checkpointing to this model?
