# Implement Top-k and Top-p Sampling

**Category:** coding
**Difficulty:** 3
**Tags:** coding, sampling, inference, generation

## Question
Implement top-k and top-p (nucleus) sampling for text generation.

Your implementation should:
- Support top-k sampling (keep only top k highest probability tokens)
- Support top-p sampling (keep tokens whose cumulative probability >= p)
- Apply temperature scaling
- Handle edge cases

**Function signature:**
```python
from typing import Optional
import torch

def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> int:
    """
    Sample next token from logits with optional top-k/top-p filtering.

    Args:
        logits: shape (vocab_size,) - raw logits for next token
        temperature: float, controls randomness (higher = more random)
        top_k: int or None, keep only top k tokens
        top_p: float or None, keep tokens with cumulative prob >= p
    Returns:
        sampled token index (int)
    """
    pass
```

## Answer

**Key concepts:**
1. Temperature scaling before softmax
2. Top-k: sort and keep top k tokens
3. Top-p: sort, compute cumulative prob, keep until threshold
4. Sample from filtered distribution

**Reference implementation:**
```python
from typing import Optional
import torch
import torch.nn.functional as F

def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> int:
    # Apply temperature scaling
    logits = logits / temperature

    # Apply top-k filtering
    if top_k is not None:
        # Get top k values and indices
        top_k = min(top_k, logits.size(-1))  # Can't be larger than vocab
        top_k_logits, top_k_indices = torch.topk(logits, top_k)

        # Zero out all logits except top k
        logits_filtered = torch.full_like(logits, float('-inf'))
        logits_filtered[top_k_indices] = top_k_logits
        logits = logits_filtered

    # Apply top-p (nucleus) filtering
    if top_p is not None:
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)

        # Compute cumulative probabilities
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff: first position where cumsum > top_p
        # Keep at least one token
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[0] = False  # Always keep the highest prob token

        # Shift right to keep the first token that exceeds top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        # Zero out removed indices
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # Sample from the filtered distribution
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token.item()

# Alternative: Combined with both filters
def sample_next_token_v2(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> int:
    # Apply temperature
    logits = logits / max(temperature, 1e-10)  # Avoid division by zero

    # Top-k filtering
    if top_k is not None and top_k > 0:
        indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    # Top-p filtering
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')

    # Sample
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
```

**Testing:**
```python
# Test with sample logits
vocab_size = 1000
logits = torch.randn(vocab_size)

# Greedy (temperature=0 approximation with top_k=1)
token_greedy = sample_next_token(logits, temperature=0.1, top_k=1)
print(f"Greedy token: {token_greedy}")
print(f"Is argmax: {token_greedy == logits.argmax().item()}")

# High temperature (more random)
tokens_high_temp = [sample_next_token(logits, temperature=2.0) for _ in range(10)]
print(f"\nHigh temp samples (diverse): {len(set(tokens_high_temp))} unique")

# Low temperature (more deterministic)
tokens_low_temp = [sample_next_token(logits, temperature=0.5) for _ in range(10)]
print(f"Low temp samples (focused): {len(set(tokens_low_temp))} unique")

# Top-k sampling
tokens_topk = [sample_next_token(logits, top_k=10) for _ in range(100)]
unique_tokens = set(tokens_topk)
print(f"\nTop-k=10: {len(unique_tokens)} unique tokens (should be <= 10)")

# Top-p sampling
tokens_topp = [sample_next_token(logits, top_p=0.9) for _ in range(100)]
print(f"Top-p=0.9: {len(set(tokens_topp))} unique tokens")

# Combined
token_combined = sample_next_token(logits, temperature=0.8, top_k=50, top_p=0.95)
print(f"\nCombined sampling: {token_combined}")
```

**Common mistakes:**
1. ❌ Not handling temperature=0 case
2. ❌ Wrong cumulative probability logic for top-p
3. ❌ Not keeping at least one token in top-p
4. ❌ Applying top-k/top-p before temperature scaling

## Follow-up Questions
- When would you use top-k vs top-p?
- How does temperature affect the distribution?
- What happens with temperature=0? With temperature approaching infinity?
