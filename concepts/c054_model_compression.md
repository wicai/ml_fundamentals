# Model Compression Techniques

**Category:** modern_llm
**Difficulty:** 3
**Tags:** compression, efficiency, deployment

## Question
What are the main techniques for compressing LLMs for deployment?

## Answer
**Goal**: Reduce model size/compute while preserving quality.

**Main techniques:**

**1. Quantization** (covered separately)
- 16-bit → 8-bit → 4-bit weights
- 4× memory reduction
- Minimal quality loss with good techniques (GPTQ, AWQ)

**2. Pruning**
```
Remove unnecessary weights/neurons

Structured pruning: Remove entire channels/heads
  - 32 attention heads → prune to 24 heads
  - Preserve hardware efficiency

Unstructured pruning: Remove individual weights
  - Set small weights to zero
  - Requires sparse matrix support
```

**Magnitude pruning:**
```
1. Train full model
2. Remove weights with |w| < threshold
3. Fine-tune remaining weights
4. Repeat (iterative pruning)
```

**Typical results:**
- 30% sparsity: Minimal quality loss
- 50% sparsity: Some degradation
- 70%+ sparsity: Significant quality loss

**3. Distillation**
```
Train small "student" model to mimic large "teacher"

Loss = α * task_loss + (1-α) * distillation_loss

where:
  distillation_loss = KL(P_student || P_teacher)
```

**Example:**
- Teacher: GPT-3 175B
- Student: GPT-3 1.3B
- Student trained on teacher's outputs
- Result: ~100× smaller, 90-95% quality

**Variants:**
- **Task-specific**: Distill for one task
- **Task-agnostic**: Preserve general capabilities
- **Self-distillation**: Distill into same architecture (surprisingly helps)

**4. Architecture optimization**

**Grouped-Query Attention (GQA):**
- Share K,V across attention heads
- 4-8× less KV cache memory
- Minimal quality loss

**Mixture of Experts (MoE):**
- Conditional computation (not all params active)
- Larger total params, same inference cost

**5. Weight sharing**
```
ALBERT: Share weights across layers
  Instead of 24 unique layers, repeat same layer 24 times
  24× fewer parameters
  Quality loss, but helps for smaller models
```

**6. Kernel fusion & operator optimization**
```
Flash Attention, fused layernorm+residual
Same model, 2-3× faster inference
No quality change
```

**Combining techniques:**
```
Start: Llama-7B (14GB in fp16)
  → LoRA fine-tune (custom behavior)
  → Quantize to 4-bit (3.5GB)
  → Prune 20% weights (2.8GB)
  → Deploy with optimized kernels

Result: 5× smaller, 2× faster, 95% quality
```

**Trade-offs:**

| Technique | Size | Speed | Quality | Training |
|-----------|------|-------|---------|----------|
| Quantization | ↓↓ | ↑ | ↓ | None |
| Pruning | ↓ | ↑ | ↓ | Fine-tune |
| Distillation | ↓↓↓ | ↑↑ | ↓↓ | Full train |
| Architecture | ↓ | ↑ | ↓ | Retrain |

**Modern examples:**
- **DistilBERT**: 60% of BERT size, 95% quality
- **TinyLlama**: 1B distilled model, decent quality
- **Phi-2**: 2.7B model trained on quality data, beats 7B models

## Follow-up Questions
- What's the difference between pruning and quantization?
- How does knowledge distillation work?
- Can you combine quantization + pruning?
