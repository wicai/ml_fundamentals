# Attention Score Gradient Derivation

**Category:** fundamentals
**Difficulty:** 5
**Tags:** derivation, math, attention, backpropagation

## Problem
Derive the gradient of attention scores with respect to queries, keys, and values from first principles.

## Instructions
- Start from the attention formula: Attention(Q, K, V) = softmax(QK^T / √d_k)V
- Derive ∂L/∂Q, ∂L/∂K, and ∂L/∂V
- Show all intermediate steps using chain rule
- Explain the intuition at each step
- Verify dimensions match at each step
- Highlight key insights about gradient flow

## Tools
Use pen and paper. Work through the math carefully.
