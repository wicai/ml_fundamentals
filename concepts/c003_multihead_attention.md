# Multi-Head Attention

**Category:** transformers
**Difficulty:** 3
**Tags:** attention, architecture

## Question
Why use multiple attention heads instead of one large attention mechanism?

## Answer
Multi-head attention runs multiple attention mechanisms in parallel with different learned projections.

**Architecture:**
```
For h heads with model dim d_model:
- Each head has dimension d_k = d_model / h
- Project Q, K, V to d_k dimensions (different weights per head)
- Compute attention for each head independently
- Concatenate outputs and project back to d_model
```

**Benefits:**

1. **Different relationships**: Each head can learn different patterns
   - Head 1: syntactic relationships (subject-verb)
   - Head 2: semantic similarity
   - Head 3: positional proximity

2. **Ensemble effect**: Multiple hypotheses averaged together

3. **Computational efficiency**: 8 heads × (d/8)² = d²/8 vs 1 head × d²
   - Same total parameters but more expressive

**Typical config**: GPT-3 uses 96 heads with d_model=12288, so d_k=128 per head

## Follow-up Questions
- What happens if you use just one head?
- How do you choose the number of heads?
