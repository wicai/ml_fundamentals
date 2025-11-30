# Mixed Precision Training

**Category:** training
**Difficulty:** 2
**Tags:** optimization, efficiency

## Question
What's the difference between FP16 and BF16 for mixed precision training?

## Answer
**FP16**: 5 exponent bits, 10 mantissa → small range, needs gradient scaling
**BF16**: 8 exponent bits, 7 mantissa → same range as FP32, no gradient scaling

BF16 is preferred for LLM training (simpler, more stable).
