# Transformer Complexity

**Category:** transformers
**Difficulty:** 3
**Tags:** complexity, attention

## Question
What is the time complexity of standard self-attention with respect to sequence length n?

## Answer
**O(n²)** for both time and memory.

Computing QK^T creates an n×n attention matrix. This quadratic scaling is why long sequences (>8K tokens) are expensive.

Flash Attention reduces memory to O(n) while keeping O(n²) compute.
