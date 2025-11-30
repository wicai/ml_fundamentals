# Perplexity

**Category:** evaluation
**Difficulty:** 3
**Tags:** evaluation, metrics, training

## Question
What is perplexity and why is it used to evaluate language models?

## Answer
**Perplexity**: Measure of how "surprised" a model is by test data. Lower = better.

**Definition:**
```
PPL = exp(cross_entropy_loss)
    = exp(-1/N * Σ log P(x_i | context))
    = (1 / P(x_1, ..., x_N))^(1/N)
```

**Intuition**: Perplexity of k means "model is as confused as if it had to choose uniformly from k possibilities."

**Example:**
- Perfect model (always correct): PPL = 1
- Random guessing (vocab size 50k): PPL = 50,000
- GPT-3: PPL ≈ 20 on various benchmarks

**Why used:**

1. **Interpretable**: "Model is uncertain over ~20 tokens on average"
2. **Comparable**: Can compare across models, datasets (if same vocabulary)
3. **Training objective**: Directly corresponds to next-token prediction loss

**Connection to cross-entropy:**
```
Loss = -log P(token)
PPL = exp(loss)

If loss = 3.0, then PPL = exp(3.0) ≈ 20
```

**Limitations:**

1. **Not task performance**: Low PPL ≠ good downstream performance
   - Model might predict common words well but fail at reasoning
2. **Vocabulary dependent**: Different tokenizers → incomparable PPL
3. **Distribution dependent**: PPL on web text ≠ PPL on books

**Modern perspective:**
- **Training**: Track loss (cross-entropy), easier to interpret gradients
- **Evaluation**: Track PPL for interpretability
- **Deployment**: Track task-specific metrics (accuracy, ROUGE, etc.)

**Scaling law connection:**
PPL improves as power law with compute, model size, data size.

## Follow-up Questions
- How does perplexity relate to entropy?
- Can you compare perplexity across different tokenizers?
- What's a good perplexity value?
