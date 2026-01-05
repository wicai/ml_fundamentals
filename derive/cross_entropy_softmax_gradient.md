# Cross-Entropy + Softmax Gradient

**Category:** fundamentals
**Difficulty:** 4
**Tags:** derivation, math, loss-functions, backpropagation

## Problem
Derive the gradient of cross-entropy loss combined with softmax activation. Show that it simplifies to (p - y).

## Instructions
- Start from L = -∑ y_i log(p_i) where p = softmax(z)
- Derive ∂L/∂z_j for each component
- Show all intermediate steps using chain rule
- Prove the elegant simplification: ∂L/∂z = p - y
- Explain why this simplification is significant
- Verify dimensions match

## Tools
Use pen and paper. Work through the math carefully.
