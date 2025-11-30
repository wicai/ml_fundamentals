# Batch Size Effects

**Category:** training
**Difficulty:** 3
**Tags:** training, optimization, hyperparameters

## Question
How does batch size affect training? What are the trade-offs of large vs small batches?

## Answer
**Batch size impact:**

**Small batches (1-32):**
- ✓ More gradient noise → better exploration → better generalization
- ✓ Less memory
- ✗ Slower training (less parallelism)
- ✗ Noisier convergence
- ✗ Might need more iterations

**Large batches (256-4096+):**
- ✓ Faster training (GPU utilization, fewer updates)
- ✓ Stabler convergence
- ✓ Better for distributed training
- ✗ More memory
- ✗ Worse generalization ("sharp minima")
- ✗ Requires learning rate tuning

**Linear scaling rule:**
If you increase batch size by k, increase learning rate by k.
- Batch 256 with lr=0.001
- Batch 2048 with lr=0.008
- Rationale: Effective gradient is averaged over batch, so larger batch = less noise = can take bigger steps

**Modern LLM training:**
- Start small (256-512) for stability
- Gradually increase to 2048-4096 for efficiency
- Use gradient accumulation to simulate large batch

**Gradient noise scale:**
```
B_simple = (tr(H) / ||g||²) * N

where:
  H = Hessian
  g = gradient
  N = dataset size

If batch_size > B_simple: wasting compute (diminishing returns)
```

**Critical batch size**: Point where doubling batch size doesn't reduce training time.

**Gotchas:**
- Batch size affects batch normalization statistics (not relevant for transformers with layer norm)
- Very large batches need careful warmup and LR tuning

## Follow-up Questions
- Why do large batches generalize worse?
- How does gradient accumulation compare to large batches?
- What's the linear scaling rule and when does it break?
