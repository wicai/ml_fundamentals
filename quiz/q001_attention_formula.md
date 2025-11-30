# Attention Formula

**Category:** transformers
**Difficulty:** 3
**Tags:** attention, formula

## Question
What is the formula for scaled dot-product attention?

## Answer
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

where d_k = dimension of keys
```

The √d_k scaling prevents dot products from becoming too large (which would push softmax into saturated regions with tiny gradients).
