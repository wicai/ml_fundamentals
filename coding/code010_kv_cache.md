# Implement KV Cache for Autoregressive Generation

**Category:** coding
**Difficulty:** 4
**Tags:** coding, optimization, inference, attention

## Question
Implement KV caching for efficient autoregressive text generation.

Your implementation should:
- Cache key and value tensors from previous tokens
- Reuse cached KV for new token generation
- Support batched generation
- Avoid recomputing attention for past tokens

**Function signature:**
```python
from typing import Optional, Tuple
import torch
import torch.nn as nn

class AttentionWithKVCache(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        """
        Args:
            d_model: model dimension
            num_heads: number of attention heads
        """
        pass

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: input tokens, shape (batch, seq_len, d_model)
            kv_cache: optional tuple of (cached_keys, cached_values)
                      each of shape (batch, num_heads, prev_seq_len, d_k)
            use_cache: whether to return updated cache
        Returns:
            output: shape (batch, seq_len, d_model)
            new_kv_cache: tuple of (keys, values) if use_cache else None
        """
        pass
```

## Answer

**Key concepts:**
1. Cache K and V from previous tokens
2. Compute Q only for new tokens
3. Concatenate cached KV with new KV
4. Attention over all previous + current tokens

**Reference implementation:**
```python
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionWithKVCache(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)"""
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)"""
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.size()

        # Compute Q for current tokens (always needed)
        Q = self.W_q(x)
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, d_k)

        # Compute K and V for current tokens
        K_new = self.W_k(x)
        V_new = self.W_v(x)
        K_new = self.split_heads(K_new)
        V_new = self.split_heads(V_new)

        # If we have cached KV, concatenate with new KV
        if kv_cache is not None:
            K_cached, V_cached = kv_cache
            K = torch.cat([K_cached, K_new], dim=2)  # Concat along seq_len
            V = torch.cat([V_cached, V_new], dim=2)
        else:
            K = K_new
            V = V_new

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(self.d_k, dtype=scores.dtype))

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        # Combine heads and project
        output = self.combine_heads(output)
        output = self.W_o(output)

        # Return cache if requested
        new_cache = (K, V) if use_cache else None

        return output, new_cache

# Example usage in generation
from typing import List

def generate_with_cache(
    model: AttentionWithKVCache,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int = 10
) -> List[torch.Tensor]:
    """
    Generate text using KV caching.

    Args:
        model: AttentionWithKVCache module
        prompt_tokens: initial tokens, shape (batch, prompt_len, d_model)
        max_new_tokens: number of tokens to generate
    Returns:
        all_outputs: list of output tensors
    """
    batch_size = prompt_tokens.size(0)
    kv_cache = None
    all_outputs = []

    # Process prompt (can cache here too)
    output, kv_cache = model(prompt_tokens, kv_cache=None, use_cache=True)
    all_outputs.append(output)

    # Generate new tokens one at a time
    current_token = output[:, -1:, :]  # Last token's output

    for _ in range(max_new_tokens):
        # Only process the new token (seq_len=1)
        # Reuse cached KV from all previous tokens
        output, kv_cache = model(current_token, kv_cache=kv_cache, use_cache=True)
        all_outputs.append(output)

        current_token = output  # Use this as input for next iteration

    return all_outputs
```

**Testing:**
```python
# Test
batch_size, d_model, num_heads = 2, 512, 8
model = AttentionWithKVCache(d_model, num_heads)

# Prompt tokens
prompt = torch.randn(batch_size, 5, d_model)

# Without cache (baseline)
print("=== Without KV Cache ===")
full_seq = prompt
for i in range(3):
    output_no_cache, _ = model(full_seq, kv_cache=None, use_cache=False)
    next_token = torch.randn(batch_size, 1, d_model)
    full_seq = torch.cat([full_seq, next_token], dim=1)
    print(f"Step {i}: Processing {full_seq.size(1)} tokens")

# With cache (efficient)
print("\n=== With KV Cache ===")
kv_cache = None
current = prompt
for i in range(3):
    output_cache, kv_cache = model(current, kv_cache=kv_cache, use_cache=True)
    next_token = torch.randn(batch_size, 1, d_model)
    current = next_token  # Only process new token

    cache_seq_len = kv_cache[0].size(2) if kv_cache else 0
    print(f"Step {i}: Processing 1 token, cache has {cache_seq_len} tokens")

# Verify correctness
print("\n=== Correctness Check ===")
# Process sequence all at once
full_seq = torch.randn(batch_size, 10, d_model)
output_full, _ = model(full_seq, kv_cache=None, use_cache=False)

# Process incrementally with cache
kv_cache = None
output_last = None
for i in range(10):
    token = full_seq[:, i:i+1, :]
    output, kv_cache = model(token, kv_cache=kv_cache, use_cache=True)
    output_last = output

# Last token output should match
diff = (output_full[:, -1:, :] - output_last).abs().max()
print(f"Difference in last token: {diff.item():.2e}")
print(f"Outputs match: {diff.item() < 1e-5}")
```

**Common mistakes:**
1. ❌ Not concatenating along the correct dimension
2. ❌ Recomputing K and V for cached tokens
3. ❌ Wrong cache shape handling for batches
4. ❌ Not updating cache with new tokens

## Follow-up Questions
- What's the memory vs computation tradeoff?
- How much speedup does KV caching provide?
- How would you implement this with different cache sizes per layer?
