# Cross-Entropy Loss

**Category:** foundations
**Difficulty:** 3
**Tags:** loss, optimization, fundamentals

## Question
Why is cross-entropy the standard loss function for language modeling?

## Answer
**Cross-Entropy Loss:**
```
L = -Σ y_i * log(p_i)

For language modeling (single correct token):
L = -log p_correct

where p = softmax(logits)
```

**Why cross-entropy for classification/LM:**

1. **Probabilistic interpretation**: Minimizing cross-entropy = maximizing likelihood
   - Want to maximize P(correct token)
   - Equivalent to minimizing -log P(correct token)

2. **Convex in logits**: Easy to optimize (gradient always points toward correct answer)

3. **Infinite gradient at wrong answer**: Strongly penalizes confident wrong predictions
   - log(0.01) = -4.6 (confident wrong)
   - log(0.5) = -0.69 (uncertain)

4. **Information theory**: Measures KL divergence between predicted and true distribution

**Gradient (key for interviews):**
```
Softmax: p_i = exp(z_i) / Σ exp(z_j)
Cross-entropy: L = -log p_y (y = correct class)

∂L/∂z_i = p_i - 1[i=y]

In words: "predicted probability - 1 if correct class, else 0"
```

**Example:**
```
True label: y = 2 (third class)
Logits: [1.0, 2.0, 3.0]
Softmax: [0.09, 0.24, 0.67]
Loss: -log(0.67) = 0.4

Gradient: [0.09, 0.24, 0.67-1] = [0.09, 0.24, -0.33]
```

**vs Mean Squared Error:**
- MSE doesn't penalize confident mistakes strongly enough
- Cross-entropy has better gradients for classification

## Follow-up Questions
- What's the gradient of softmax + cross-entropy?
- Why not use MSE for classification?
- How does label smoothing modify cross-entropy?
