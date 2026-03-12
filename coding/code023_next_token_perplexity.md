# Compute Next-Token Perplexity

**Category:** coding
**Difficulty:** 3
**Tags:** coding, evaluation, broadcasting, tensor-manipulation, interview-prep

## Question
Implement a function that computes deterministic next-token perplexity on a batch of tokenized texts, given a model that returns logits.

Your implementation should:
- Take pre-tokenized input_ids and a model that returns logits
- Correctly align logits with shifted labels for next-token prediction
- Handle variable-length sequences with padding (pad_token_id given)
- Create a padding mask and broadcast it correctly
- Return per-sequence perplexity and batch-average perplexity

**Function signature:**
```python
import torch
import torch.nn.functional as F

def compute_perplexity(
    input_ids: torch.Tensor,
    model,
    pad_token_id: int = 0
) -> dict:
    """
    Compute next-token perplexity for a batch of sequences.

    Args:
        input_ids: shape (batch_size, seq_len) - tokenized input sequences, right-padded
        model: callable that takes input_ids and returns logits of shape (batch_size, seq_len, vocab_size)
        pad_token_id: token id used for padding

    Returns:
        dict with:
            - 'per_sequence_perplexity': shape (batch_size,) - perplexity for each sequence
            - 'mean_perplexity': scalar - average perplexity across sequences
    """
    pass
```

**Hints / things to watch out for:**
1. Logits at position `t` predict the token at position `t+1` — you need to shift correctly
2. The padding mask must be built for the *labels* (shifted), not the inputs
3. When computing mean log-prob per sequence, you must divide by the number of *non-padded* tokens in each sequence — broadcasting the mask correctly is critical
4. `torch.gather` or advanced indexing to select the logit corresponding to each target token

## Answer

**Key concepts:**
1. Shift logits and labels so logits[t] predicts labels[t] = input_ids[t+1]
2. Build a mask for non-padding positions in the shifted labels
3. Compute log-probs via log-softmax, gather target token log-probs
4. Mask out padding positions, sum per-sequence, divide by token counts (broadcasting)
5. Perplexity = exp(-mean_log_prob)

**Reference implementation:**
```python
import torch
import torch.nn.functional as F

def compute_perplexity(
    input_ids: torch.Tensor,
    model,
    pad_token_id: int = 0
) -> dict:
    # Get model logits: (batch_size, seq_len, vocab_size)
    with torch.no_grad():
        logits = model(input_ids)

    # Shift: logits[:-1] predict input_ids[1:]
    # shift_logits: (batch_size, seq_len - 1, vocab_size)
    # shift_labels: (batch_size, seq_len - 1)
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # Log-softmax over vocab dimension: (batch_size, seq_len - 1, vocab_size)
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather the log-prob of the actual next token at each position
    # shift_labels.unsqueeze(-1): (batch_size, seq_len - 1, 1)
    # After gather: (batch_size, seq_len - 1, 1) -> squeeze -> (batch_size, seq_len - 1)
    token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    # Mask: only count non-padding positions in the shifted labels
    # mask: (batch_size, seq_len - 1), float
    mask = (shift_labels != pad_token_id).float()

    # Zero out log-probs at padding positions
    # Broadcasting: token_log_probs (batch_size, seq_len-1) * mask (batch_size, seq_len-1)
    # element-wise, no broadcasting needed here — but the division below is key
    masked_log_probs = token_log_probs * mask

    # Sum log-probs per sequence, divide by number of real tokens per sequence
    # sum over seq dim: (batch_size,)
    sum_log_probs = masked_log_probs.sum(dim=-1)
    # token counts per sequence: (batch_size,)
    token_counts = mask.sum(dim=-1)

    # Mean log-prob per sequence: (batch_size,)
    # Avoid division by zero for fully-padded sequences
    mean_log_probs = sum_log_probs / token_counts.clamp(min=1)

    # Perplexity = exp(-mean_log_prob)
    per_sequence_perplexity = torch.exp(-mean_log_probs)

    return {
        'per_sequence_perplexity': per_sequence_perplexity,
        'mean_perplexity': per_sequence_perplexity.mean()
    }
```

**Testing:**
```python
import torch

# Dummy model: returns random logits
class DummyLM:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
    def __call__(self, input_ids):
        batch_size, seq_len = input_ids.shape
        return torch.randn(batch_size, seq_len, self.vocab_size)

# Perfect model: always assigns prob 1.0 to the correct next token
class PerfectLM:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
    def __call__(self, input_ids):
        batch_size, seq_len = input_ids.shape
        # Create logits that strongly predict the next token
        logits = torch.full((batch_size, seq_len, self.vocab_size), -100.0)
        # For positions 0..seq_len-2, set high logit for input_ids at position 1..seq_len-1
        for t in range(seq_len - 1):
            for b in range(batch_size):
                logits[b, t, input_ids[b, t + 1]] = 100.0
        return logits

vocab_size = 100
pad_token_id = 0

# Test 1: Perfect model should give perplexity ~1.0
input_ids = torch.tensor([
    [5, 10, 15, 20, 25],
    [3, 7, 11, 0, 0],   # padded sequence
])
result = compute_perplexity(input_ids, PerfectLM(vocab_size), pad_token_id=pad_token_id)
print(f"Perfect model perplexity (should be ~1.0): {result['per_sequence_perplexity']}")

# Test 2: Random model should give perplexity ~vocab_size
result_random = compute_perplexity(input_ids, DummyLM(vocab_size), pad_token_id=pad_token_id)
print(f"Random model perplexity (should be ~{vocab_size}): {result_random['per_sequence_perplexity']}")

# Test 3: Verify padded tokens are excluded
# Sequence 2 has only 2 real prediction positions (3->7, 7->11), not 4
print(f"Mean perplexity: {result['mean_perplexity']:.4f}")
```

**Common mistakes:**
1. ❌ Not shifting logits/labels — using logits[t] to predict input_ids[t] instead of input_ids[t+1]
2. ❌ Building the padding mask on input_ids instead of the shifted labels
3. ❌ Forgetting to divide by per-sequence token counts (using a global mean instead)
4. ❌ Off-by-one: the mask and shifted logits must have the same seq_len dimension (seq_len - 1)
5. ❌ Using softmax then log instead of log_softmax (numerical instability)
6. ❌ Forgetting `unsqueeze(-1)` when using `gather` on the vocab dimension

## Follow-up Questions
- How would you modify this for a causal model that uses a key-value cache (computing perplexity token-by-token)?
- How would you handle a case where the first token is a BOS token that shouldn't be predicted?
- What's the relationship between perplexity and bits-per-byte? How would you convert?
- How would you compute perplexity on a dataset that doesn't fit in memory (streaming)?
