# Data Parallelism

**Category:** training
**Difficulty:** 3
**Tags:** distributed_training, parallelism, optimization

## Question
What is data parallelism and how is it implemented for distributed training?

## What to Cover
- **Set context by**: Explaining DP as the simplest distributed training approach (replicate model, split data)
- **Must mention**: The algorithm (micro-batches, all-reduce gradients, synchronous update), DDP vs FSDP, communication overhead
- **Show depth by**: Discussing ring all-reduce efficiency, sync vs async variants, and when model parallelism is needed instead
- **Avoid**: Skipping the all-reduce communication cost or conflating DP with DDP

## Answer
**Data Parallelism (DP)**: Replicate model on multiple GPUs, split data across GPUs.

**Algorithm:**
```
1. Each GPU has full copy of model
2. Split batch across GPUs (micro-batches)
3. Each GPU computes gradients on its micro-batch
4. Synchronize gradients (all-reduce)
5. Each GPU updates its model copy (now identical)
```

**Example:**
- 8 GPUs, batch size 256
- Each GPU: micro-batch of 32
- Forward/backward in parallel
- Synchronize gradients: all-reduce operation
- Update weights

**Communication:**
- **Naive**: Each GPU sends gradients to all others → O(n²) communication
- **All-Reduce (Ring)**: O(n) communication
  - Pass gradients in ring, each GPU accumulates
  - More efficient for large n

**Variants:**

**1. Distributed Data Parallel (DDP) - PyTorch**
```python
model = torch.nn.parallel.DistributedDataParallel(model)
```
- Each process owns one GPU
- Gradients all-reduced via NCCL
- Standard for multi-node training

**2. Fully Sharded Data Parallel (FSDP) / ZeRO**
- Shard model parameters across GPUs
- Each GPU only stores subset of parameters
- Gather parameters during forward/backward
- Enables much larger models
- Covered separately

**Gradient synchronization strategies:**

**Synchronous (standard):**
- Wait for all GPUs before updating
- Guarantees identical models
- Straggler problem (slow GPU blocks all)

**Asynchronous:**
- Update immediately without waiting
- Faster but models diverge
- Rarely used for LLMs

**Bandwidth bottleneck:**
- Gradient size = model size
- 7B model (fp16) = 14GB gradients to synchronize
- Needs fast interconnect (InfiniBand, NVLink)

**When to use:**
- Model fits in single GPU memory → DP
- Model doesn't fit → need model parallelism too

## Follow-up Questions
- What's the communication overhead of data parallelism?
- How does gradient accumulation interact with data parallelism?
- What's the difference between DP and DDP?
