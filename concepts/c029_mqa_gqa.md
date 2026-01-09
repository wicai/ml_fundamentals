# Multi-Query & Grouped-Query Attention

**Category:** modern_llm
**Difficulty:** 4
**Tags:** attention, architecture, inference

## Question
What are MQA and GQA? Why are they used in modern LLMs?

## What to Cover
- **Set context by**: Explaining the KV cache memory bottleneck that MQA/GQA solve
- **Must mention**: How each works (MQA: 1 KV head, GQA: grouped KV heads), memory savings, quality tradeoff (MHA > GQA > MQA, but GQA is close to MHA)
- **Show depth by**: Giving adoption examples (LLaMA 1 vs 2, Falcon) and the implementation detail (repeat_interleave)
- **Avoid**: Describing the mechanisms without explaining *why* they matter (inference batch size scaling)

## Answer
**Standard Multi-Head Attention:**
```
For h heads:
  Each head has separate Q, K, V projections
  K_i, V_i have shape [seq_len, d_model/h]
  KV cache: 2 * h * seq_len * d_model/h = 2 * seq_len * d_model
```

**Multi-Query Attention (MQA):**
```
All heads share single K, V
  Q has h heads (separate projections)
  K, V have 1 head (shared across all Q heads)
  KV cache: 2 * seq_len * head_dim (h× smaller, where head_dim = d_model/h)
```

**Grouped-Query Attention (GQA):**
```
Heads divided into groups, each group shares K, V
  Q has h heads
  K, V have h/g heads (g = group size)
  Example: 32 Q heads, 8 KV heads → 4 Q heads per KV head
  KV cache: 2 * (h/g) * seq_len * d_model/h
```

**Comparison:**
| Method | # KV heads | KV cache size | Quality |
|--------|-----------|---------------|---------|
| MHA | h | 2 × seq × d | Best |
| GQA | h/g | 2 × seq × d / g | Good |
| MQA | 1 | 2 × seq × d / h | Acceptable |

**Why this matters:**

**Inference bottleneck**: KV cache limits batch size
- GPT-3 (96 heads): 96× KV memory
- MQA version: 1× KV memory → 96× larger batch size!

**Training speed**: Slightly faster (fewer K,V computations)

**Quality trade-off:**
- MHA > GQA > MQA
- But GQA is very close to MHA (surprising!)

**Modern adoption:**
- **LLaMA 1**: MHA (32 heads)
- **LLaMA 2**: GQA (32 Q heads, 8 KV heads)
- **Falcon**: MQA
- **GPT-4**: Unknown (likely GQA)

**Implementation:**
```python
# GQA attention
n_q_heads = 32
n_kv_heads = 8
heads_per_group = n_q_heads // n_kv_heads  # 4

Q = Q.view(batch, seq, n_q_heads, head_dim)
K = K.view(batch, seq, n_kv_heads, head_dim)
V = V.view(batch, seq, n_kv_heads, head_dim)

# Repeat K,V for each group
K = K.repeat_interleave(heads_per_group, dim=2)
V = V.repeat_interleave(heads_per_group, dim=2)

# Now standard MHA
```

## Follow-up Questions
- How much memory does GQA save for LLaMA-70B?
- Why doesn't MQA hurt quality more?
- Can you convert MHA to GQA post-training?
