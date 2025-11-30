# RoPE vs Absolute Positional Encoding

**Category:** transformers
**Difficulty:** 3
**Tags:** position, architecture

## Question
What's the main advantage of RoPE over learned absolute positional embeddings?

## Answer
**Length generalization**: RoPE can handle sequences longer than those seen during training.

RoPE encodes **relative** positions (distance between tokens) rather than absolute positions, so it can extrapolate. Learned absolute embeddings have fixed maximum length.

Example: Train on 2K tokens → RoPE can infer on 8K+ tokens (with some degradation).
