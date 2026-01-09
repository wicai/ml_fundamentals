# Pipeline Parallelism

**Category:** training
**Difficulty:** 4
**Tags:** distributed_training, parallelism, architecture

## Question
What is pipeline parallelism and how does it complement other parallelism strategies?

## What to Cover
- **Set context by**: Explaining PP splits model by layers across GPUs (vs TP which splits within layers)
- **Must mention**: The pipeline bubble problem, micro-batching to reduce bubble, GPipe vs 1F1B schedules
- **Show depth by**: Giving a hybrid 3D parallelism example (TP + PP + DP) and explaining when PP is appropriate
- **Avoid**: Describing naive pipelining without mentioning the bubble problem and solutions

## Answer
**Pipeline Parallelism (PP)**: Split model layers across GPUs, pipeline the micro-batches.

**Idea**: GPU 1 processes layers 1-8, GPU 2 processes layers 9-16, etc.

**Naive pipeline (bad):**
```
GPU 1: [Forward layers 1-8] → idle → idle → [Backward layers 1-8]
GPU 2: idle → [Forward layers 9-16] → idle → [Backward layers 9-16]

Problem: 75% idle time (pipeline bubble)!
```

**GPipe (better):**
```
Split batch into m micro-batches
GPU 1: F1 → F2 → F3 → ... → B3 → B2 → B1
GPU 2:      F1 → F2 → F3 → ... → B3 → B2 → B1

Bubble reduced to O(num_stages / num_microbatches)
```

**PipeDream / 1F1B (best):**
- "One Forward, One Backward" schedule
- Interleave forward and backward passes
- Minimal bubble
- More memory efficient than GPipe

**Memory trade-off:**
- **GPipe**: Store all micro-batch activations before backward
  - High memory, low bubble
- **1F1B**: Stream through micro-batches
  - Lower memory, low bubble

**Example (4-stage pipeline, 8 micro-batches):**
```
Time:  0  1  2  3  4  5  6  7  8  9  10 11 12
GPU0: F0 F1 F2 F3 F4 B0 B1 B2 B3 B4
GPU1:    F0 F1 F2 F3 F4 B0 B1 B2 B3 B4
GPU2:       F0 F1 F2 F3 F4 B0 B1 B2 B3 B4
GPU3:          F0 F1 F2 F3 F4 B0 B1 B2 B3

Bubble: First 3 and last 3 time steps
Efficiency: ~75% (vs 25% naive)
```

**Communication:**
- Send activations forward
- Send gradients backward
- Point-to-point (not all-reduce)

**Hybrid parallelism (3D):**
```
GPT-3 training (1024 GPUs):
  8-way TP (within node)
  16-way PP (across nodes)
  8-way DP (data batches)

Total: 8 × 16 × 8 = 1024 GPUs
```

**Trade-offs:**

✓ Enables training very large models
✓ Less communication than TP (only between adjacent stages)
✗ Pipeline bubble (wasted compute)
✗ Complexity (gradient accumulation, scheduling)

**When to use:**
- Very large models (100B+)
- Combine with TP and DP
- Need to split across many nodes

## Follow-up Questions
- What's the pipeline bubble and how do you minimize it?
- How does PP compare to TP in terms of communication?
- What's the 1F1B schedule?
