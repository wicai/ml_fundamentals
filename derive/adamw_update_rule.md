# AdamW Update Rule Derivation

**Category:** fundamentals
**Difficulty:** 4
**Tags:** derivation, math, optimization, adam

## Problem
Derive the AdamW optimizer update rule from first principles, showing how it differs from standard Adam.

## Instructions
- Start from Adam's update rule with momentum and RMSprop
- Show the first moment estimate (m_t) and second moment estimate (v_t)
- Derive the bias correction terms
- Show how weight decay is decoupled in AdamW vs L2 regularization in Adam
- Explain the final update rule: θ_t = θ_{t-1} - η * (m̂_t / √v̂_t + wd * θ_{t-1})
- Explain intuition for why decoupled weight decay is better
- Verify dimensions match

## Tools
Use pen and paper. Work through the math carefully.
