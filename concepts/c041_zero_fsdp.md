# ZeRO & FSDP

**Category:** training
**Difficulty:** 4
**Tags:** distributed_training, memory, optimization

## Question
What is ZeRO/FSDP and how does it reduce memory usage for training?

## Answer
**Problem**: Data parallelism replicates model on each GPU → wasted memory.

**Memory breakdown (per GPU in data parallel):**
```
Model parameters: 7B × 2 bytes (fp16) = 14GB
Gradients: 14GB (same size as params)
Optimizer states (Adam): 28GB (m and v, fp32)
Activations: 10-100GB (depends on batch size)

Total: ~66GB minimum (without activations!)
```

**ZeRO (Zero Redundancy Optimizer)** - DeepSpeed

**ZeRO Stage 1**: Partition optimizer states
- Each GPU stores optimizer states for 1/N of parameters
- Save 4× memory (optimizer states biggest component)

**ZeRO Stage 2**: Partition optimizer states + gradients
- Also shard gradients across GPUs
- Save ~8× memory

**ZeRO Stage 3**: Partition everything (params + grads + optimizer)
- Each GPU only stores 1/N of model
- Gather parameters when needed for forward/backward
- Save ~N× memory (N = number of GPUs)

**Algorithm (ZeRO-3):**
```
1. Forward pass:
   - All-gather parameters for current layer
   - Compute forward
   - Discard parameters

2. Backward pass:
   - All-gather parameters for current layer
   - Compute backward
   - Reduce-scatter gradients (each GPU gets subset)
   - Discard parameters

3. Optimizer step:
   - Update local parameter shard with local gradients
```

**FSDP** (Fully Sharded Data Parallel) - PyTorch
- PyTorch's implementation of ZeRO-3
- Similar algorithm, different API
- Standard in PyTorch 2.0+

**Memory savings example:**
```
70B model on 8 GPUs:
  Standard DP: 70B params per GPU (doesn't fit!)
  FSDP: 70B / 8 = 8.75B params per GPU (fits!)
```

**Communication cost:**
- All-gather parameters: 2× model size per layer
- More communication than standard DP
- But enables training much larger models

**CPU offloading (ZeRO-Infinity):**
- Offload optimizer states to CPU
- Offload parameters to CPU or NVMe
- Train 1T+ models on limited GPUs

**Trade-offs:**

✓ Train much larger models with same hardware
✓ Same convergence as standard DP (mathematically equivalent)
✗ More communication overhead
✗ Can be slower for small models that fit in memory

**When to use:**
- Model doesn't fit in GPU with standard DP
- Want to increase batch size

## Follow-up Questions
- What's the communication pattern of FSDP?
- How does FSDP compare to tensor parallelism?
- What's ZeRO-Infinity?
