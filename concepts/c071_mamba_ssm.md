# Mamba & State Space Models

**Category:** modern_llm
**Difficulty:** 4
**Tags:** architecture, efficiency, recurrent

## Question
What is Mamba and how do state space models differ from transformers?

## Answer
**State Space Models (SSMs)**: Alternative to attention using recurrent state updates.

**Problem with transformers:**
- O(n²) attention complexity
- O(n) memory for KV cache during inference
- Limits long sequences

**SSM approach:**
```
Instead of attention over all tokens:
  Maintain fixed-size state h
  Update state recurrently

h_t = A * h_{t-1} + B * x_t
y_t = C * h_t

Like RNN but with specific structure
```

**S4 (Structured State Spaces):**

**Key innovation**: Diagonal state matrix for efficiency
```
Standard SSM: A is n×n matrix (expensive)
S4: A is diagonal (cheap)

Allows parallel training via convolution form
O(n log n) training, O(n) inference
```

**Mamba (2023):**

**Improvements over S4:**

**1. Selective state spaces:**
```
Standard SSM: A, B, C are fixed (data-independent)
Mamba: A, B, C are functions of input (data-dependent)

B_t = Linear(x_t)  # Input-dependent

Allows model to decide what to remember/forget
```

**2. Hardware-aware implementation:**
```
Fused CUDA kernels
Avoids materializing large states
Similar philosophy to Flash Attention
```

**3. Simplified architecture:**
```
No attention, no MLP
Just: SSM + gating

Mamba Block:
  x → LayerNorm → SSM → Gate → output
  Residual connection
```

**Comparison to transformers:**

| Aspect | Transformer | Mamba |
|--------|-------------|-------|
| Training | O(n²) | O(n) |
| Inference | O(n) per token + KV cache | O(1) per token |
| Memory | O(n) KV cache | O(1) state |
| Context | All previous tokens | Fixed-size state |
| Long sequences | Expensive | Efficient |

**Advantages of Mamba:**

1. **Constant memory inference**: State size doesn't grow with sequence
2. **Fast inference**: O(1) per token (no KV cache)
3. **Long context**: Train on 1M+ tokens efficiently
4. **Efficient training**: Linear in sequence length

**Disadvantages:**

1. **In-context learning**: Weaker than transformers
2. **Copying**: Harder to copy exact sequences
3. **Random access**: Can't attend to specific token directly
4. **Maturity**: Less research/tooling than transformers

**Performance:**

**Language modeling:**
- Mamba-3B matches transformer-3B
- Mamba-7B competitive with Llama 7B on some tasks
- Still slightly behind on aggregate benchmarks

**Long-range tasks:**
- Mamba excels (audio, DNA sequences, long documents)
- Transformers struggle or need sparse attention

**Hybrid models:**

**Idea**: Combine attention and SSM
```
Layer 1: Attention
Layer 2: Mamba
Layer 3: Attention
...

Get benefits of both:
  - Attention for in-context learning
  - SSM for efficiency
```

**Examples:**
- **Jamba (AI21)**: Hybrid Mamba + Attention
- **StripedHyena**: Hybrid architecture

**Training Mamba:**

**Parallel form (training):**
```
Compute as convolution:
  y = Conv(x, K)

where K is derived from A, B, C

Allows parallel processing like transformers
```

**Recurrent form (inference):**
```
h_t = A*h_{t-1} + B*x_t
y_t = C*h_t

Sequential, but O(1) per token
```

**Use cases:**

✓ **Long sequences**: DNA, audio, long documents
✓ **Low-latency inference**: Constant memory
✓ **Streaming**: Natural for real-time processing

✗ **Few-shot learning**: Weaker than transformers
✗ **RAG**: Transformers better with explicit retrieval

**Modern perspective (2024):**

- Mamba shows promise but hasn't replaced transformers
- Hybrid models likely future
- Active research area

## Follow-up Questions
- Why are state space models more efficient than attention?
- What is selective state space?
- When would you choose Mamba over a transformer?
