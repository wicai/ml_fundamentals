# Perplexity Definition

**Category:** evaluation
**Difficulty:** 2
**Tags:** metrics, evaluation

## Question
How is perplexity related to cross-entropy loss?

## Answer
```
Perplexity = exp(cross_entropy_loss)
          = exp(-1/N * Σ log P(x_i | context))
```

If loss = 3.0, then perplexity = exp(3) ≈ 20 (model is "confused" over ~20 options on average).
