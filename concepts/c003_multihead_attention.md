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

1. **Different representation subspaces** (Vaswani et al., 2017):
   - "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions"
   - Each head can learn different patterns:
     - Head 1: syntactic relationships (subject-verb)
     - Head 2: semantic similarity
     - Head 3: positional proximity
   - Single head averaging would inhibit this diversity

2. **Task specialization during training** (Li et al., 2024):
   - Gradient flow naturally causes "task allocation" where each head focuses on different aspects
   - Creates diverse gradient paths that improve learning dynamics
   - Multi-head models significantly outperform single-head for multi-task scenarios

3. **Parameter efficiency**:
   - Total parameters: 3d² (same as single head projecting to d_model)
   - But split across h heads, each with dimension d_k = d_model/h
   - QK^T computation: O(seq_len² × d) same as single head
   - The benefit is representation diversity, NOT reduced computation

**Typical config**: GPT-3 uses 96 heads with d_model=12288, so d_k=128 per head

## Follow-up Questions
- What happens if you use just one head?
- How do you choose the number of heads?
