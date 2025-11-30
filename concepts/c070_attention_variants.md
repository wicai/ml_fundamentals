# Attention Variants & Efficient Attention

**Category:** transformers
**Difficulty:** 4
**Tags:** attention, efficiency, architecture

## Question
What are the main approaches to making attention more efficient than O(n²)?

## Answer
**Problem**: Standard attention is O(n²) in sequence length.
```
For n=8K: 64M attention scores
For n=128K: 16B attention scores (250× more!)
```

**Main efficiency approaches:**

**1. Sparse Attention (Longformer, BigBird)**

**Idea**: Each token attends to subset of tokens, not all.

**Patterns:**
```
Local attention: Attend to nearby tokens (±k window)
  Token i attends to [i-k, i+k]
  O(n*k) complexity

Global attention: Few tokens attend to all
  [CLS] token attends globally
  O(n + g*n) where g = global tokens

Random attention: Random subset of tokens
  Helps connectivity
```

**Longformer:**
```
Combine:
  - Local (window=512)
  - Global (task tokens)
  - Dilated (every kth token)

Result: O(n) attention, works on 4K-16K sequences
```

**2. Linformer**

**Idea**: Project K, V to lower dimension before attention.
```
Standard: Q(n × d) @ K^T(d × n) = (n × n)

Linformer: Q(n × d) @ [K(n × d) → K'(k × d)]^T = (n × k)

where k << n (e.g., k=256, n=4096)

Complexity: O(n*k) instead of O(n²)
```

**3. Performer / Linear Attention**

**Idea**: Approximate softmax with kernel trick.
```
Standard: softmax(QK^T)V

Performer: ϕ(Q) @ (ϕ(K)^T @ V)

where ϕ = feature map (e.g., random Fourier features)

Complexity: O(n*d²) - linear in sequence length!
```

**4. Flash Attention** (covered separately)

**Idea**: Not lower complexity, but memory-efficient.
- Still O(n²) FLOPs
- O(n) memory instead of O(n²)
- Fused kernel, no materialization of attention matrix
- 2-4× faster in practice

**5. Memory-compressed attention (Compressive Transformer)**

**Idea**: Compress past into summary.
```
Recent tokens: Full attention (1024 tokens)
Older tokens: Compressed representation (10× compression)
Very old: Further compressed or discarded

Sliding window with compression
```

**6. Sliding window attention**

**Idea**: Only attend to recent k tokens.
```
Token i attends to [i-k, i]

Simple, O(n*k)
Used in some code models (k=2048)
```

**7. Multi-query / Grouped-query attention**

**Not lower complexity, but faster inference:**
- Share K,V across heads
- Reduces KV cache size
- Faster memory access

**8. Recurrent / State Space Models (Mamba, RWKV)**

**Idea**: Replace attention with recurrent mechanism.
```
No attention at all!
O(n) for training and inference

Trade-off: Weaker long-range dependencies
```

**Comparison:**

| Method | Train complexity | Inference | Quality loss |
|--------|------------------|-----------|--------------|
| Standard | O(n²) | O(n²) | None |
| Flash Attention | O(n²) | O(n²) | None |
| Longformer | O(n) | O(n) | Small |
| Linformer | O(n) | O(n) | Medium |
| Performer | O(n) | O(n) | Medium |
| Mamba | O(n) | O(n) | Small |

**Which to use:**

**Flash Attention:** Always (free speedup, no quality loss)

**Longformer/BigBird:** Long documents (16K+ tokens), need full context

**Mamba/RWKV:** Very long sequences, recurrent inference

**Linformer/Performer:** Experimental, less adoption

**Sliding window:** Code/simple sequential tasks

**Modern trend:**
- Flash Attention is standard
- Most models still use full attention with Flash
- Sparse attention for specialized long-context models
- State space models (Mamba) gaining traction

**Limitations:**

- Most methods have quality trade-offs
- Full attention still best for most tasks
- Hardware optimized for dense ops, not sparse

## Follow-up Questions
- Why hasn't sparse attention replaced standard attention?
- What is the kernel trick in Performer?
- How does Flash Attention achieve O(n) memory?
