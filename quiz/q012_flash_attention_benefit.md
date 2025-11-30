# Flash Attention Benefit

**Category:** modern_llm
**Difficulty:** 2
**Tags:** optimization, attention

## Question
What is the main benefit of Flash Attention?

## Answer
**Memory efficiency**: Reduces memory from O(n²) to O(n) for the attention mechanism.

It achieves this by not materializing the full attention matrix, instead computing in fused kernels with tiling. Also 2-4× faster due to reduced memory I/O.
