# Quantization for LLMs

**Category:** modern_llm
**Difficulty:** 4
**Tags:** inference, optimization, compression

## Question
What is quantization and how is it applied to LLMs? What are the main approaches?

## Answer
**Quantization**: Represent weights/activations in lower precision (int8, int4) instead of fp16/fp32.

**Why quantize:**
- **Memory**: 4-bit = 4× memory reduction → LLaMA-70B fits in 40GB vs 160GB
- **Speed**: int8 math is faster on some hardware
- **Cost**: Smaller models = cheaper inference

**Main approaches:**

**1. Post-Training Quantization (PTQ)**
No retraining, just convert weights after training.

- **Zero-shot**: Simple rounding (poor quality)
- **Calibration-based**: Use small dataset to find optimal scales

**2. Quantization-Aware Training (QAT)**
Simulate quantization during training.

- Better quality
- Much more expensive (full training run)

**Quantization schemes:**

**Symmetric:**
```
x_quant = round(x / scale)
scale = max(abs(x)) / (2^(bits-1) - 1)
```

**Asymmetric:**
```
x_quant = round((x - zero_point) / scale)
Find scale and zero_point to minimize error
```

**Modern techniques:**

**GPTQ (Post-Training):**
- Layer-wise quantization with Hessian information
- 4-bit weights, minimal quality loss
- Fast inference

**AWQ (Activation-aware Weight Quantization):**
- Protect important weights (high activation magnitude)
- 4-bit, better than GPTQ on some tasks

**GGML/GGUF (llama.cpp):**
- Mixed precision (2-6 bits per layer)
- Optimized for CPU inference
- Extremely popular for local models

**Typical configurations:**

| Method | Bits | Quality | Speed | Memory |
|--------|------|---------|-------|--------|
| FP16 | 16 | Baseline | 1× | 1× |
| INT8 | 8 | 99% | 1.5× | 0.5× |
| INT4 | 4 | 95% | 2× | 0.25× |

**Gotcha**: Quantization affects different models differently. Math/reasoning tasks more sensitive than chat.

## Follow-up Questions
- How does quantization affect model quality?
- What's the difference between weight-only and activation quantization?
- Can you quantize below 4 bits?
