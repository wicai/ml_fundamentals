# Extract Token-Level Log-Probabilities

**Category:** coding
**Difficulty:** 3
**Tags:** coding, inference, log-probs, language model, safety

## Question
Given a language model and a token sequence, extract per-token log-probabilities and compute sequence-level scores. This is foundational for reward modeling, DPO, PPO surrogate objectives, perplexity, and safety evaluations.

Your implementation should include:
1. **`get_per_token_logprobs`**: Extract log P(token_t | tokens_<t) for each token
2. **`sequence_logprob`**: Sum per-token log-probs to get sequence-level log-probability
3. **`sequence_perplexity`**: Compute perplexity from per-token log-probs
4. **`compare_sequences`**: Given a prompt and two completions, which does the model prefer?

**Function signature:**
```python
def get_per_token_logprobs(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Extract per-token log-probabilities from a causal language model.

    For each position t, compute log P(input_ids[t] | input_ids[0:t]).
    Position 0 has no conditioning, so we only return positions 1..T-1.

    Args:
        model: causal LM that returns logits of shape (batch, seq_len, vocab_size)
        input_ids: token IDs, shape (batch_size, seq_len)
    Returns:
        token_logprobs: shape (batch_size, seq_len - 1)
            token_logprobs[b, t] = log P(input_ids[b, t+1] | input_ids[b, 0:t+1])
    """
    pass

def sequence_logprob(token_logprobs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute sequence-level log-probability by summing per-token log-probs.

    Args:
        token_logprobs: shape (batch_size, seq_len)
        mask: optional bool mask, shape (batch_size, seq_len). True = include token.
              Use this to mask out prompt tokens (only score the completion).
    Returns:
        seq_logprobs: shape (batch_size,)
    """
    pass

def sequence_perplexity(token_logprobs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute per-sequence perplexity: exp(-1/N * sum(log_probs)).

    Args:
        token_logprobs: shape (batch_size, seq_len)
        mask: optional bool mask, shape (batch_size, seq_len)
    Returns:
        perplexity: shape (batch_size,)
    """
    pass

def compare_sequences(model: nn.Module, prompt_ids: torch.Tensor, completion_a_ids: torch.Tensor, completion_b_ids: torch.Tensor) -> dict[str, float]:
    """
    Given a prompt and two completions, compute which the model assigns higher probability.

    Args:
        model: causal LM
        prompt_ids: shape (1, prompt_len)
        completion_a_ids: shape (1, completion_a_len)
        completion_b_ids: shape (1, completion_b_len)
    Returns:
        dict with 'logprob_a', 'logprob_b', 'perplexity_a', 'perplexity_b', 'preferred' ('a' or 'b')
    """
    pass
```

## Answer

**Key concepts:**
1. A causal LM predicts P(next_token | previous_tokens) at each position
2. Logits at position t predict the token at position t+1 (shifted by one)
3. Sequence log-prob = sum of per-token log-probs
4. Perplexity = exp(-avg log-prob) — lower is better
5. For scoring completions, only sum log-probs over completion tokens (not the prompt)

**Reference implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_per_token_logprobs(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Extract per-token log-probs from a causal LM."""
    with torch.no_grad():
        logits = model(input_ids)  # (batch, seq_len, vocab_size)

    # Shift: logits[t] predicts input_ids[t+1]
    shift_logits = logits[:, :-1, :]      # (batch, seq_len-1, vocab_size)
    shift_labels = input_ids[:, 1:]        # (batch, seq_len-1)

    # Compute log-softmax over vocabulary
    log_probs = F.log_softmax(shift_logits, dim=-1)  # (batch, seq_len-1, vocab_size)

    # Gather the log-prob of the actual next token
    token_logprobs = log_probs.gather(
        dim=2,
        index=shift_labels.unsqueeze(-1),  # (batch, seq_len-1, 1)
    ).squeeze(-1)  # (batch, seq_len-1)

    return token_logprobs

def sequence_logprob(token_logprobs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Sum per-token log-probs, optionally masked."""
    if mask is not None:
        return (token_logprobs * mask).sum(dim=-1)
    return token_logprobs.sum(dim=-1)

def sequence_perplexity(token_logprobs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Perplexity = exp(-avg log-prob)."""
    if mask is not None:
        n_tokens = mask.sum(dim=-1)
        total_logprob = (token_logprobs * mask).sum(dim=-1)
    else:
        n_tokens = torch.tensor(token_logprobs.shape[-1], dtype=torch.float)
        total_logprob = token_logprobs.sum(dim=-1)

    avg_logprob = total_logprob / n_tokens
    return torch.exp(-avg_logprob)

def compare_sequences(model: nn.Module, prompt_ids: torch.Tensor, completion_a_ids: torch.Tensor, completion_b_ids: torch.Tensor) -> dict[str, float]:
    """Compare two completions under the model."""
    prompt_len = prompt_ids.shape[1]

    # Concatenate prompt + completion for each
    full_a = torch.cat([prompt_ids, completion_a_ids], dim=1)
    full_b = torch.cat([prompt_ids, completion_b_ids], dim=1)

    # Get per-token log-probs
    logprobs_a = get_per_token_logprobs(model, full_a)
    logprobs_b = get_per_token_logprobs(model, full_b)

    # Create masks: only score completion tokens (not prompt)
    # logprobs are shifted by 1, so prompt tokens occupy positions 0..prompt_len-2
    # completion tokens start at position prompt_len-1
    mask_a = torch.zeros_like(logprobs_a, dtype=torch.bool)
    mask_a[:, prompt_len - 1:] = True
    mask_b = torch.zeros_like(logprobs_b, dtype=torch.bool)
    mask_b[:, prompt_len - 1:] = True

    lp_a = sequence_logprob(logprobs_a, mask_a.float()).item()
    lp_b = sequence_logprob(logprobs_b, mask_b.float()).item()
    ppl_a = sequence_perplexity(logprobs_a, mask_a.float()).item()
    ppl_b = sequence_perplexity(logprobs_b, mask_b.float()).item()

    return {
        'logprob_a': lp_a,
        'logprob_b': lp_b,
        'perplexity_a': ppl_a,
        'perplexity_b': ppl_b,
        'preferred': 'a' if lp_a > lp_b else 'b',
    }
```

**Testing:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# Create a simple "language model" (just a linear layer for testing)
vocab_size = 50
seq_len = 10

class SimpleLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.linear(self.embed(input_ids))

model = SimpleLM(vocab_size)
model.eval()

# Test 1: Per-token log-probs shape and values
print("=" * 70)
print("TEST 1: Per-Token Log-Probs")
print("=" * 70)
input_ids = torch.randint(0, vocab_size, (2, seq_len))
token_lps = get_per_token_logprobs(model, input_ids)
print(f"Input shape: {input_ids.shape}")
print(f"Log-probs shape: {token_lps.shape} (should be (2, {seq_len-1}))")
print(f"All log-probs <= 0: {(token_lps <= 0).all().item()}")
print(f"Sample log-probs: {token_lps[0, :5].tolist()}")

# Test 2: Sequence log-prob
print("\n" + "=" * 70)
print("TEST 2: Sequence Log-Prob")
print("=" * 70)
seq_lp = sequence_logprob(token_lps)
print(f"Sequence log-probs: {seq_lp.tolist()}")
print(f"All negative: {(seq_lp < 0).all().item()}")
manual = token_lps.sum(dim=-1)
print(f"Matches manual sum: {torch.allclose(seq_lp, manual)}")

# Test 3: Masked sequence log-prob (only score last 5 tokens)
print("\n" + "=" * 70)
print("TEST 3: Masked Log-Prob (Completion Only)")
print("=" * 70)
mask = torch.zeros_like(token_lps)
mask[:, 4:] = 1.0  # Only score positions 4 onward
masked_lp = sequence_logprob(token_lps, mask)
manual_masked = (token_lps[:, 4:]).sum(dim=-1)
print(f"Masked log-prob: {masked_lp.tolist()}")
print(f"Manual masked:   {manual_masked.tolist()}")
print(f"Match: {torch.allclose(masked_lp, manual_masked)}")

# Test 4: Perplexity
print("\n" + "=" * 70)
print("TEST 4: Perplexity")
print("=" * 70)
ppl = sequence_perplexity(token_lps)
print(f"Perplexity: {ppl.tolist()}")
print(f"All >= 1: {(ppl >= 1).all().item()}")
# For random model, perplexity should be near vocab_size
print(f"Expected ~{vocab_size} for random model")

# Test 5: Compare sequences
print("\n" + "=" * 70)
print("TEST 5: Compare Sequences")
print("=" * 70)
prompt = torch.randint(0, vocab_size, (1, 3))
comp_a = torch.randint(0, vocab_size, (1, 4))
comp_b = torch.randint(0, vocab_size, (1, 4))
result = compare_sequences(model, prompt, comp_a, comp_b)
print(f"Completion A log-prob: {result['logprob_a']:.4f}")
print(f"Completion B log-prob: {result['logprob_b']:.4f}")
print(f"Model prefers: {result['preferred']}")

# Test 6: Verify against manual cross-entropy
print("\n" + "=" * 70)
print("TEST 6: Verify Against F.cross_entropy")
print("=" * 70)
input_ids = torch.randint(0, vocab_size, (1, 8))
token_lps = get_per_token_logprobs(model, input_ids)
avg_nll = -token_lps.mean()

logits = model(input_ids)
ce = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab_size), input_ids[:, 1:].reshape(-1))
print(f"Avg NLL from log-probs: {avg_nll.item():.6f}")
print(f"F.cross_entropy:        {ce.item():.6f}")
print(f"Match: {torch.isclose(avg_nll, ce)}")
```

**Common mistakes:**
1. Forgetting the shift: logits[t] predicts token[t+1], not token[t]
2. Using softmax instead of log_softmax (then taking log — loses precision)
3. Not masking out prompt tokens when scoring completions
4. Confusing sequence log-prob (sum) with average log-prob (mean)
5. Computing perplexity as exp(-sum) instead of exp(-mean)

## Follow-up Questions
- Why do we shift logits by one position?
- When would you use length-normalized log-prob vs raw log-prob?
- How are per-token log-probs used in the PPO objective for RLHF?
- How does this relate to the DPO loss (which uses policy vs reference log-probs)?
- What is the relationship between cross-entropy loss and perplexity?
