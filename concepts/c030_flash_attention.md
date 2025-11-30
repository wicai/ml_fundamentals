# Flash Attention

**Category:** modern_llm
**Difficulty:** 4
**Tags:** attention, optimization, efficiency

## Question
What problem does Flash Attention solve and how does it work at a high level?

## Answer
**Problem**: Standard attention is memory-bound, not compute-bound.

**Standard attention:**
```
1. Compute S = QK^T (materialize n×n matrix)
2. Compute P = softmax(S) (materialize n×n matrix)
3. Compute O = PV (n×d output)

Memory: O(n²) for S and P matrices
```

For n=2048, d=128: S and P use 2048² = 4M values each!

**Flash Attention key ideas:**

1. **Tiling/Blocking**: Split Q, K, V into blocks
2. **Fused kernel**: Compute attention in one GPU kernel (no intermediate materialization)
3. **Recomputation**: Recompute S and P in backward pass instead of storing

**Algorithm (simplified):**
```
1. Load block of Q into fast memory (SRAM)
2. Load block of K, V into fast memory
3. Compute attention for this block
4. Write output to HBM
5. Repeat for all blocks
```

**Key insight**: Trading recomputation (cheap, lots of GPU FLOPs) for memory I/O (expensive bottleneck).

**Benefits:**

- **Memory**: O(n) instead of O(n²)
  - Can train with much longer sequences
  - Less memory for activations
- **Speed**: 2-4× faster (less HBM access)
  - Training: 15% end-to-end speedup
  - Inference: Bigger speedup for long sequences

**Requirements:**
- CUDA/Triton implementation (architecture-specific)
- Careful numerical stability (softmax in blocks)

**Flash Attention 2:**
- Further optimizations (parallelism, work partitioning)
- 2× faster than Flash Attention 1

**Adoption:**
- Standard in PyTorch 2.0+ (torch.nn.functional.scaled_dot_product_attention)
- Used in training LLaMA 2, GPT-4, etc.

**Gotcha**: Implementation complexity is high, but usage is simple (drop-in replacement for standard attention).

## Follow-up Questions
- Why is standard attention memory-bound not compute-bound?
- How does Flash Attention handle softmax numerics?
- What's the speedup for different sequence lengths?
