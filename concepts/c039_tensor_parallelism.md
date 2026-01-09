# Tensor Parallelism

**Category:** training
**Difficulty:** 4
**Tags:** distributed_training, parallelism, architecture

## Question
What is tensor parallelism and when is it needed for training LLMs?

## What to Cover
- **Set context by**: Explaining TP is for when model doesn't fit in single GPU memory
- **Must mention**: How layers are split (column-parallel, row-parallel), communication pattern (2 all-reduces per block), comparison with DP
- **Show depth by**: Discussing hybrid TP+DP configurations (TP within node, DP across nodes) and interconnect requirements (NVLink)
- **Avoid**: Only describing the concept without explaining the communication overhead or when to use TP vs other parallelism

## Answer
**Tensor Parallelism (TP)**: Split individual layers across multiple GPUs.

**Problem**: Model too large to fit in single GPU memory.

**How it works (Megatron-LM style):**

**Example: Linear layer Y = XW**
```
Standard: W is [d_in, d_out], stored on 1 GPU

TP (column-parallel):
  Split W into [W_1, W_2] along output dimension
  GPU 1: Y_1 = X @ W_1
  GPU 2: Y_2 = X @ W_2
  Concatenate: Y = [Y_1, Y_2]
```

**Transformer attention:**
```
Q, K, V projections: Column-parallel split
  Each GPU computes subset of heads

Output projection: Row-parallel split
  Each GPU has partial result
  All-reduce to get final output
```

**Feed-forward network:**
```
First layer (expand 4x): Column-parallel
Second layer (project back): Row-parallel
```

**Communication:**
- All-reduce after row-parallel layers
- No communication for column-parallel layers
- Total: 2 all-reduces per transformer block

**Memory savings:**
- TP across n GPUs → ~n× less memory per GPU
- Example: 70B model, 4-way TP → ~18B params per GPU

**vs Data Parallelism:**

| Aspect | Data Parallel | Tensor Parallel |
|--------|---------------|-----------------|
| Model size | Fits on 1 GPU | Doesn't fit on 1 GPU |
| Replication | Full model replicas | Model sharded |
| Communication | Gradients | Activations + gradients |
| Granularity | Batch level | Layer level |

**Hybrid: TP + DP**
```
Example: 512 GPUs training 175B model
  8-way TP (within node, fast interconnect)
  64-way DP (across nodes)
  Effective batch: 64 × local_batch
```

**Trade-offs:**

✓ Enables training models larger than single GPU
✗ More communication overhead (activations)
✗ Requires fast interconnect (NVLink within node)

**Modern trend:**
- TP within nodes (8 GPUs with NVLink)
- DP across nodes
- FSDP/ZeRO for memory efficiency

## Follow-up Questions
- How do you split attention across GPUs?
- What's the communication cost of tensor parallelism?
- When should you use TP vs FSDP?
