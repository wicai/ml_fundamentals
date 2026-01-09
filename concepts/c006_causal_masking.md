# Causal/Autoregressive Masking

**Category:** transformers
**Difficulty:** 3
**Tags:** attention, architecture, autoregressive

## Question
What is causal masking and why is it necessary for language models?

## What to Cover
- **Set context by**: Explaining the training=inference consistency requirement for autoregressive models
- **Must mention**: How masking is implemented (-∞ in attention scores), why it prevents information leakage, how parallel training still works
- **Show depth by**: Contrasting with bidirectional models (BERT) that don't need causal masking
- **Avoid**: Describing only the mechanics without explaining *why* it's necessary

## Answer
Causal masking prevents positions from attending to future positions during self-attention.

**Implementation:**
```
# Before softmax:
scores = QK^T / √d_k
mask = upper_triangular_matrix(-inf)  # Future positions = -∞
scores = scores + mask
attention = softmax(scores)  # Future positions get probability ≈ 0
```

**Why Necessary:**

1. **Training = inference**: Model can only use past context during generation, so train same way
2. **Prevent information leakage**: Without mask, token at position i could "cheat" by looking at position i+1
3. **Parallel training**: Can still train on full sequence at once (teacher forcing), just mask attention

**Example:**
```
Input: "The cat sat on"
Target: "cat sat on the"

Position 0 (The) can attend to: [The]
Position 1 (cat) can attend to: [The, cat]
Position 2 (sat) can attend to: [The, cat, sat]
...
```

**Note**: Encoder models (BERT) don't use causal masking - they can attend bidirectionally because they're not autoregressive.

## Follow-up Questions
- How does this enable parallel training?
- What happens if you forget the mask?
- Why can BERT attend bidirectionally?
