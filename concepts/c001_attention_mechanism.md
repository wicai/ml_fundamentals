# Attention Mechanism

**Category:** transformers
**Difficulty:** 4
**Tags:** attention, architecture, fundamentals

## Question
What is the attention mechanism and why is it better than RNNs for sequence modeling?

## Answer
Attention is a mechanism that computes a weighted sum of values based on the similarity between a query and keys.

**Core Formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Why better than RNNs:**

1. **Parallelization**: Can process entire sequence at once, RNNs must process sequentially
2. **Long-range dependencies**: Direct connections between all positions (O(1) path), RNNs have O(n) path
3. **Information flow**: No bottleneck of fixed hidden state, each position can attend to all others
4. **Gradient flow**: Direct gradients between distant positions, no vanishing gradient through time

**Key tradeoff**: O(n²) complexity vs O(n) for RNNs, but GPU parallelization + sequence length usually < 2048 makes this worthwhile.

## Follow-up Questions
- What's the purpose of the √d_k scaling factor?
- How does multi-head attention extend this?
- What's the complexity for sequence length n?
