# Gradient Clipping

**Category:** training
**Difficulty:** 2
**Tags:** optimization, training

## Question
What is gradient clipping by global norm and what threshold is typical for LLMs?

## Answer
**Gradient clipping**: Limit gradient magnitude to prevent exploding gradients.

```
global_norm = sqrt(Σ ||g_i||²)
if global_norm > threshold:
    g = g * (threshold / global_norm)
```

**Typical threshold for LLMs**: 1.0
