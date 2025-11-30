# Learning Rate Schedules

**Category:** training
**Difficulty:** 3
**Tags:** training, optimization, hyperparameters

## Question
What learning rate schedule is used for LLM pretraining and why?

## Answer
**Standard LLM schedule: Cosine decay with warmup**

```
1. Warmup (0 to max_lr) - Linear increase
   lr = max_lr * (step / warmup_steps)
   Duration: 2000-10000 steps

2. Cosine decay (max_lr to min_lr)
   lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
   Duration: Rest of training

where progress = (step - warmup_steps) / (total_steps - warmup_steps)
```

**Why warmup?**

1. **Stability**: Early gradients are noisy, large LR causes divergence
2. **Adam statistics**: m and v estimates need time to stabilize
3. **Embeddings**: Randomly initialized embeddings need gentle start

**Why cosine decay?**

1. **Smooth**: Gradual decrease, no sharp drops
2. **Final convergence**: Lower LR at end helps fine-tune solution
3. **Empirically better**: Beats step decay, linear decay

**Typical values:**
- max_lr: 3e-4 to 6e-4
- min_lr: max_lr / 10 (e.g., 3e-5)
- warmup: 2000 steps or 1-5% of training
- total_steps: 100K-1M depending on dataset size

**Alternative: Constant LR with warmup**
- Some recent work shows constant LR (after warmup) works well
- Simpler, one less hyperparameter
- May need more tuning of the constant value

**Gotcha**: Learning rate is often the most important hyperparameter. Get this wrong and nothing works.

## Follow-up Questions
- Why not just use constant learning rate?
- How does this interact with gradient accumulation?
- What happens if warmup is too short or too long?
