# Compute Perplexity for Language Models

**Category:** coding
**Difficulty:** 2
**Tags:** coding, evaluation, perplexity, language model

## Question
Implement perplexity computation for evaluating language models. Perplexity measures how "surprised" a model is by text — lower perplexity means the model predicts the text better.

Your implementation should include:
1. **`compute_perplexity`**: Compute perplexity of a model on a text sequence
2. **`compute_perplexity_batched`**: Compute over a dataset with proper handling of long sequences
3. **`compare_models`**: Compare two models on the same text

**Function signature:**
```python
def compute_perplexity(model: nn.Module, input_ids: torch.Tensor) -> float:
    """
    Compute perplexity of a language model on input tokens.

    PPL = exp( -1/N * sum_{t=1}^{N} log P(token_t | token_{<t}) )

    Args:
        model: causal LM returning logits of shape (batch, seq_len, vocab_size)
        input_ids: token IDs, shape (1, seq_len)
    Returns:
        perplexity: scalar float (lower is better)
    """
    pass

def compute_perplexity_batched(
    model: nn.Module,
    input_ids: torch.Tensor,
    stride: int = 512,
    max_length: int = 1024,
) -> float:
    """
    Compute perplexity on long text using a sliding window approach.

    For texts longer than max_length, use a sliding window with overlap.
    Only count each token once (using the stride region's predictions).

    Args:
        model: causal LM
        input_ids: shape (1, total_length) — can be very long
        stride: how far to slide the window each step
        max_length: context window size of the model
    Returns:
        perplexity: scalar float
    """
    pass

def compare_models(
    model_a: nn.Module,
    model_b: nn.Module,
    input_ids: torch.Tensor,
) -> dict[str, float]:
    """
    Compare perplexity of two models on the same text.

    Args:
        model_a, model_b: two causal LMs
        input_ids: shape (1, seq_len)
    Returns:
        dict with 'ppl_a', 'ppl_b', 'better' ('a' or 'b')
    """
    pass
```

## Answer

**Key concepts:**
1. Perplexity = exp(cross-entropy) = exp(-avg log-prob per token)
2. A perfect model has perplexity 1. Random on vocab V has perplexity V.
3. For long texts, use sliding window to stay within model's context length
4. With sliding window, use stride < max_length for overlap (better predictions at boundaries)
5. Perplexity is the standard metric for comparing language models

**Reference implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_perplexity(model: nn.Module, input_ids: torch.Tensor) -> float:
    """Compute perplexity on a single sequence."""
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)  # (1, seq_len, vocab)

    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # Cross-entropy per token
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    return torch.exp(loss).item()

def compute_perplexity_batched(
    model: nn.Module,
    input_ids: torch.Tensor,
    stride: int = 512,
    max_length: int = 1024,
) -> float:
    """Sliding window perplexity for long texts."""
    model.eval()
    seq_len = input_ids.size(1)

    total_nll = 0.0
    total_tokens = 0

    # Slide window across the text
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        input_chunk = input_ids[:, begin:end]

        # For the first window, score all tokens
        # For subsequent windows, only score tokens in the stride region
        # (tokens before stride already have predictions from previous window)
        target_start = 0 if begin == 0 else max_length - stride

        with torch.no_grad():
            logits = model(input_chunk)

        # Only compute loss on the stride region
        shift_logits = logits[:, target_start:-1, :].contiguous()
        shift_labels = input_chunk[:, target_start + 1:].contiguous()

        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='sum',
        )

        total_nll += nll.item()
        total_tokens += shift_labels.numel()

        if end == seq_len:
            break

    avg_nll = total_nll / total_tokens
    return float(torch.exp(torch.tensor(avg_nll)).item())

def compare_models(
    model_a: nn.Module,
    model_b: nn.Module,
    input_ids: torch.Tensor,
) -> dict[str, float]:
    """Compare perplexity of two models."""
    ppl_a = compute_perplexity(model_a, input_ids)
    ppl_b = compute_perplexity(model_b, input_ids)

    return {
        'ppl_a': ppl_a,
        'ppl_b': ppl_b,
        'better': 'a' if ppl_a < ppl_b else 'b',  # Lower is better
    }
```

**Testing:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# Simple LM for testing
class SimpleLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.embed(x))

vocab_size = 100
model = SimpleLM(vocab_size)
model.eval()

# Test 1: Basic perplexity
print("=" * 70)
print("TEST 1: Basic Perplexity")
print("=" * 70)
input_ids = torch.randint(0, vocab_size, (1, 50))
ppl = compute_perplexity(model, input_ids)
print(f"Perplexity: {ppl:.2f}")
print(f"Expected ~{vocab_size} for random model")
print(f"Perplexity >= 1: {ppl >= 1.0}")

# Test 2: Verify against manual computation
print("\n" + "=" * 70)
print("TEST 2: Verify Against Manual")
print("=" * 70)
with torch.no_grad():
    logits = model(input_ids)
shift_logits = logits[:, :-1, :]
shift_labels = input_ids[:, 1:]
ce = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
ppl_manual = torch.exp(ce).item()
print(f"Our perplexity: {ppl:.4f}")
print(f"Manual:         {ppl_manual:.4f}")
print(f"Match: {abs(ppl - ppl_manual) < 1e-3}")

# Test 3: Better model = lower perplexity
print("\n" + "=" * 70)
print("TEST 3: Better Model = Lower Perplexity")
print("=" * 70)

# Train model briefly on the test sequence to lower perplexity
trained_model = SimpleLM(vocab_size)
optimizer = torch.optim.Adam(trained_model.parameters(), lr=0.01)
for _ in range(100):
    logits = trained_model(input_ids)
    loss = F.cross_entropy(logits[:, :-1, :].view(-1, vocab_size), input_ids[:, 1:].view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

trained_model.eval()
ppl_untrained = compute_perplexity(model, input_ids)
ppl_trained = compute_perplexity(trained_model, input_ids)
print(f"Untrained PPL: {ppl_untrained:.2f}")
print(f"Trained PPL:   {ppl_trained:.2f}")
print(f"Trained is better: {ppl_trained < ppl_untrained}")

# Test 4: Sliding window
print("\n" + "=" * 70)
print("TEST 4: Sliding Window (Long Sequence)")
print("=" * 70)
long_ids = torch.randint(0, vocab_size, (1, 200))
ppl_full = compute_perplexity(model, long_ids)
ppl_windowed = compute_perplexity_batched(model, long_ids, stride=50, max_length=100)
print(f"Full sequence PPL: {ppl_full:.2f}")
print(f"Windowed PPL:      {ppl_windowed:.2f}")
print(f"Both reasonable: {1 < ppl_full < 500 and 1 < ppl_windowed < 500}")

# Test 5: Compare models
print("\n" + "=" * 70)
print("TEST 5: Compare Models")
print("=" * 70)
result = compare_models(model, trained_model, input_ids)
print(f"Model A PPL: {result['ppl_a']:.2f}")
print(f"Model B PPL: {result['ppl_b']:.2f}")
print(f"Better: {result['better']} (should be 'b' — the trained one)")
```

**Common mistakes:**
1. Forgetting the shift (logits[t] predicts token[t+1])
2. Using `reduction='mean'` when you need per-token NLL for sliding window
3. Computing exp(mean(losses)) instead of exp(sum(losses)/N) — same thing, but be careful with batching
4. Not handling the overlap correctly in sliding window
5. Comparing perplexities across different tokenizers (incomparable!)

## Follow-up Questions
- Why is perplexity better than raw loss for comparing models?
- Why can't you compare perplexity across different tokenizers?
- What is bits-per-character (BPC) and how does it relate to perplexity?
- How is perplexity used to measure safety/capability trade-offs in alignment?
- A model has perplexity 15 on English text. What does that mean intuitively?
