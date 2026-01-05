# Backpropagation Through Transformer Block

**Category:** fundamentals
**Difficulty:** 5
**Tags:** derivation, math, transformer, backpropagation

## Problem
Derive the complete backpropagation equations through a transformer block (attention + FFN + residual + layer norm).

## Instructions
- Start from the transformer block structure:
  - x₁ = LayerNorm(x + Attention(x))
  - x₂ = LayerNorm(x₁ + FFN(x₁))
- Derive gradients flowing backward through each component
- Show how gradients split at residual connections
- Account for layer norm, attention, and FFN gradients
- Show all intermediate steps using chain rule
- Explain gradient flow intuition (why residuals help)
- Verify dimensions match at each step
- Highlight key insights about vanishing/exploding gradients

## Tools
Use pen and paper. Work through the math carefully.
