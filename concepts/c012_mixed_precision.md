# Mixed Precision Training

**Category:** training
**Difficulty:** 3
**Tags:** training, optimization, efficiency

## Question
What is mixed precision training (FP16/BF16) and why is it important for LLMs?

## What to Cover
- **Set context by**: Explaining the memory/compute constraints that make 16-bit necessary for large models
- **Must mention**: The difference between FP16 and BF16 (range vs precision tradeoff), why FP32 master weights are kept, gradient scaling for FP16
- **Show depth by**: Explaining which operations stay in FP32 (layer norm, softmax) and why BF16 is now standard
- **Avoid**: Conflating FP16 and BF16 or missing the gradient scaling requirement for FP16

## Answer
**Mixed Precision**: Use lower precision (16-bit) for most operations, higher precision (32-bit) where needed.

**Two 16-bit formats:**

**FP16 (Float16):**
- 1 sign, 5 exponent, 10 mantissa bits
- Range: ~6e-8 to 65,504
- Issue: Small range causes underflow (gradients → 0)

**BF16 (Brain Float16):**
- 1 sign, 8 exponent, 7 mantissa bits
- Range: Same as FP32 (~1e-38 to 3e38)
- Issue: Lower precision (fewer mantissa bits)

**Standard approach:**
```
1. Weights stored in FP32 (master copy)
2. Forward pass: Cast to FP16/BF16
3. Backward pass: Gradients in FP16/BF16
4. Gradient scaling (FP16 only): Scale up to prevent underflow
5. Update FP32 weights
```

**Benefits:**

1. **Memory**: 2x less memory for activations (can fit larger models/batches)
2. **Speed**: 2-3x faster on modern GPUs (Tensor Cores)
3. **Accuracy**: Minimal loss with BF16, needs tuning with FP16

**Modern trend: BF16 is standard**
- No gradient scaling needed
- Simpler implementation
- Same range as FP32
- Supported on A100, H100 GPUs

**Gotcha**: Some operations are unstable in FP16 (layer norm, softmax) - keep those in FP32.

## Follow-up Questions
- Why does FP16 need gradient scaling?
- What's the memory savings for a 70B parameter model?
- Can you train entirely in FP16 without FP32 master weights?
