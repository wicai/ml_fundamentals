# Positional Encoding

**Category:** transformers
**Difficulty:** 3
**Tags:** attention, architecture, position

## Question
Why do transformers need positional encodings and what are the main approaches?

## Answer
**Problem**: Attention is permutation-invariant - it can't distinguish between "dog bites man" and "man bites dog" without position information.

**Main Approaches:**

**1. Sinusoidal (Original Transformer)**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
- Pros: Can extrapolate to longer sequences than seen in training
- Cons: Not learned, may not be optimal

**2. Learned Absolute (GPT-2, BERT)**
- Embedding table indexed by position
- Pros: Can learn optimal encoding
- Cons: Fixed maximum length, can't extrapolate

**3. Relative (T5, Transformer-XL)**
- Encode relative distances between positions, not absolute positions
- Pros: Better length generalization
- Cons: More complex implementation

**4. Rotary (RoPE - LLaMA, GPT-NeoX)**
- Rotate Q and K embeddings based on position
- Pros: Relative encoding, excellent extrapolation, efficient
- Cons: Newer, less well understood

**Modern trend**: RoPE is becoming standard for LLMs due to length generalization.

## Follow-up Questions
- Why can't you just concatenate position as a feature?
- How does RoPE achieve relative positioning?
