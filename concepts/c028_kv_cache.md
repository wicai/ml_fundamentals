# KV Cache for Inference

**Category:** modern_llm
**Difficulty:** 4
**Tags:** inference, optimization, attention

## Question
What is KV caching and why is it critical for efficient LLM inference?

## Answer
**Problem**: Autoregressive generation recomputes attention for all previous tokens at every step.

**Without KV cache:**
```
Step 1: Compute attention for token 1
Step 2: Compute attention for tokens 1,2 (recompute token 1!)
Step 3: Compute attention for tokens 1,2,3 (recompute 1,2!)
...
```

**With KV cache:**
Store K and V matrices from previous tokens, only compute new token.

**How it works:**
```
# First token
K_1 = W_k * x_1, V_1 = W_v * x_1
cache = {K: [K_1], V: [V_1]}

# Second token
K_2 = W_k * x_2, V_2 = W_v * x_2
K = concat([cache.K, K_2])  # Reuse cached keys
V = concat([cache.V, V_2])  # Reuse cached values
Q_2 = W_q * x_2
attention = softmax(Q_2 @ K^T) @ V
cache = {K: [K_1, K_2], V: [V_1, V_2]}
```

**Benefits:**
- **Speed**: O(n) instead of O(n²) per token
  - Generate 100 tokens: 100x speedup
- **Compute**: Reduce FLOPs by ~2x

**Cost:**
- **Memory**: Store K,V for all layers, all tokens, all heads
  - LLaMA-7B, 2048 tokens: ~1GB KV cache
  - GPT-3 175B, 2048 tokens: ~40GB KV cache

**Memory formula:**
```
KV_memory = 2 * num_layers * num_heads * head_dim * seq_len * batch_size * 2_bytes

Example (LLaMA-7B):
2 * 32 layers * 32 heads * 128 head_dim * 2048 seq * 1 batch * 2 bytes
= 1GB
```

**Bottleneck**: KV cache is memory-bound, limits batch size during inference.

**Optimizations:**
- **Multi-Query Attention (MQA)**: Share K,V across heads → num_heads=1 for KV
- **Grouped-Query Attention (GQA)**: Share K,V across groups → num_kv_heads < num_q_heads
- **PagedAttention (vLLM)**: Efficient KV cache memory management

## Follow-up Questions
- How much memory does KV cache use vs model parameters?
- What is Multi-Query Attention?
- How does KV cache affect batch inference?
