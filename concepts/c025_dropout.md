# Dropout Regularization

**Category:** foundations
**Difficulty:** 3
**Tags:** regularization, training, overfitting

## Question
What is dropout and when should you use it? How does it work in transformers?

## What to Cover
- **Set context by**: Framing dropout as an ensemble/robustness regularization technique
- **Must mention**: The algorithm (mask + scale), typical values (0.1 for transformers), where applied in transformers (attention, residual, FFN)
- **Show depth by**: Noting the modern trend (large LLMs use less/no dropout), the scaling gotcha (1/(1-p))
- **Avoid**: Describing dropout without mentioning the inference-time scaling or transformer-specific placement

## Answer
**Dropout**: Randomly zero out activations during training with probability p.

**Algorithm:**
```
Training:
  mask = Bernoulli(1-p)  # Random binary mask
  output = (input * mask) / (1-p)  # Scale by 1/(1-p)

Inference:
  output = input  # No dropout, no scaling
```

**Why it works:**

1. **Ensemble effect**: Training 2^n different sub-networks (each dropout pattern)
2. **Robustness**: Forces network to not rely on any single neuron
3. **Co-adaptation**: Prevents neurons from "collaborating" too specifically

**Typical values:**
- Fully connected layers: p=0.5 (drop 50%)
- Convolutional layers: p=0.1-0.2 (lower, more parameters)
- Transformers: p=0.1 (attention + residual)

**Where dropout is applied in transformers:**

1. **Attention**: After softmax(QK^T)V
2. **Residual**: Before adding to residual stream
3. **Feed-forward**: After first linear layer (some architectures)
4. **Embeddings**: On input embeddings (less common)

**Modern trend: Less dropout**
- Large LLMs often use p=0.0-0.1 (very little)
- Other regularization: weight decay, data augmentation
- Bigger models/data = less need for dropout

**Gotcha**: Must scale by 1/(1-p) during training OR scale by (1-p) during inference. PyTorch does former (inverted dropout).

**When NOT to use:**
- Batch normalization is present (they conflict)
- Small datasets where you need full capacity

## Follow-up Questions
- Why scale by 1/(1-p)?
- How does dropout interact with batch normalization?
- Why do large LLMs use less dropout?
