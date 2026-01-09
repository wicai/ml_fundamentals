# Gradient Accumulation

**Category:** training
**Difficulty:** 2
**Tags:** training, optimization, memory

## Question
What is gradient accumulation and why is it used in LLM training?

## What to Cover
- **Set context by**: Explaining the memory constraint problem (can't fit large batch in GPU memory)
- **Must mention**: The algorithm, effective batch size formula (micro_batch × accum_steps × num_gpus), loss scaling requirement
- **Show depth by**: Noting it's mathematically equivalent to large batch and discussing wall-clock tradeoffs
- **Avoid**: Forgetting to mention the loss scaling by 1/accumulation_steps (common bug)

## Answer
**Problem**: Can't fit large batch size in GPU memory

**Solution**: Accumulate gradients over multiple forward/backward passes before updating weights.

**Algorithm:**
```
optimizer.zero_grad()
for i in range(accumulation_steps):
    outputs = model(batch_i)
    loss = loss_fn(outputs, targets_i)
    loss = loss / accumulation_steps  # Important: scale down
    loss.backward()  # Accumulates gradients
optimizer.step()  # Update weights once
```

**Effective batch size** = micro_batch × accumulation_steps × num_gpus

**Example:**
- Micro-batch per GPU: 2
- Accumulation steps: 16
- GPUs: 8
- Effective batch: 2 × 16 × 8 = 256

**Why used in LLM training:**

1. **Memory**: Large models can't fit batch_size > 1-4 per GPU
2. **Batch size matters**: Large batches (256-2048) improve training stability
3. **No extra memory**: Gradients accumulate in-place

**Trade-offs:**

✓ Same convergence as large batch (mathematically equivalent)
✓ Fits in limited memory
✗ Slower wall-clock time (less parallelism)
✗ Batch norm doesn't work (but we use Layer Norm anyway)

**Gotcha**: Must scale loss by 1/accumulation_steps or gradients will be too large!

## Follow-up Questions
- Is gradient accumulation mathematically identical to larger batch size?
- How does this interact with learning rate scheduling?
- What's the wall-clock time trade-off?
