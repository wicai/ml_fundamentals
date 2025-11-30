# Overfitting Prevention Strategies

**Category:** training
**Difficulty:** 2
**Tags:** regularization, overfitting, generalization

## Question
What are the main strategies to prevent overfitting in neural networks?

## Answer
**Overfitting**: Model memorizes training data instead of learning generalizable patterns.

**Signs:**
- Train loss keeps decreasing, validation loss increases
- Perfect train accuracy, poor test accuracy
- Model changes drastically with small data changes

**Prevention strategies:**

**1. More data**
- Most effective but expensive
- Data augmentation (images, text)
- Synthetic data generation

**2. Regularization**

**L2 (Weight decay):**
```
Loss = Task_loss + λ * ||W||²

Penalizes large weights
λ = 0.01-0.1 for LLMs
```

**L1 (Lasso):**
```
Loss = Task_loss + λ * ||W||₁

Encourages sparsity
Less common for neural nets
```

**3. Dropout**
```
Randomly zero activations with probability p
p = 0.1-0.5 depending on layer

Acts as ensemble of sub-networks
```

**4. Early stopping**
```
Monitor validation loss
Stop training when validation loss stops improving

Simple and effective
```

**5. Data augmentation**
```
Images: Rotation, flip, crop, color jitter
Text: Back-translation, synonym replacement, paraphrase
Code: Variable renaming, reordering

Creates more training examples
```

**6. Model size reduction**
```
Fewer layers, smaller width
Reduces capacity to memorize

Trade-off: Might underfit
```

**7. Batch normalization / Layer normalization**
```
Normalizes activations
Has regularizing effect (noise from batch statistics)

Standard in modern architectures
```

**8. Label smoothing**
```
Instead of: y = [0, 1, 0, 0] (one-hot)
Use: y = [0.025, 0.925, 0.025, 0.025]

Prevents overconfident predictions
Improves calibration
```

**For transformers specifically:**

1. **Pre-norm architecture**: More stable, less overfitting
2. **Low dropout** (0.1): Large models need less regularization
3. **Weight decay**: Standard (0.1)
4. **Large dataset**: Trillions of tokens
5. **Learning rate warmup**: Prevents early overfitting

**Bias-variance trade-off:**
```
Underfitting: High bias, low variance
  - Train error high
  - Simple model

Overfitting: Low bias, high variance
  - Train error low, test error high
  - Complex model

Sweet spot: Balance both
```

**When to use what:**

| Scenario | Strategy |
|----------|----------|
| Small dataset | Dropout, weight decay, early stopping |
| Large dataset | Less regularization needed |
| Deep network | Batch norm, residual connections |
| Image task | Data augmentation |
| Text task | Pretrain on more data |

**Modern LLMs:**
- Less overfitting concern (huge datasets)
- More focus on scaling, data quality
- Still use weight decay, dropout, early stopping

## Follow-up Questions
- What's the difference between L1 and L2 regularization?
- How does batch size affect overfitting?
- Can you overfit with infinite data?
