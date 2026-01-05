# Layer Normalization Forward and Backward Pass

**Category:** fundamentals
**Difficulty:** 5
**Tags:** derivation, math, normalization, backpropagation

## Problem
Derive the forward pass computation and backward pass gradients for layer normalization from first principles.

## Instructions
- Start from LayerNorm formula: y = γ * (x - μ) / σ + β
- Show forward pass: compute μ, σ², σ, normalized x, scaled output
- Derive backward pass: ∂L/∂x, ∂L/∂γ, ∂L/∂β
- Show all intermediate steps using chain rule
- Handle the mean and variance dependencies carefully
- Explain intuition at each step
- Verify dimensions match

## Tools
Use pen and paper. Work through the math carefully.
