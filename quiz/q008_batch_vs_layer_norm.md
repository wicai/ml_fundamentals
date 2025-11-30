# Batch Norm vs Layer Norm

**Category:** foundations
**Difficulty:** 3
**Tags:** normalization

## Question
What dimension does Layer Norm normalize across vs Batch Norm?

## Answer
**Batch Norm**: Normalize across **batch dimension** for each feature
- Mean/std computed over all examples in batch

**Layer Norm**: Normalize across **feature dimension** for each example
- Mean/std computed over all features per example

Transformers use Layer Norm because it's batch-size independent and works with batch=1 (inference).
