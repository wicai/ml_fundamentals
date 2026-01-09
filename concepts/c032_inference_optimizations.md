# Inference Optimizations

**Category:** modern_llm
**Difficulty:** 3
**Tags:** inference, optimization, efficiency

## Question
What are the key techniques for optimizing LLM inference latency and throughput?

## What to Cover
- **Set context by**: Distinguishing latency (user experience) from throughput (cost)
- **Must mention**: KV caching, batching (continuous batching), quantization, Flash Attention, speculative decoding, PagedAttention
- **Show depth by**: Explaining the latency breakdown (prefill vs decode) and bottleneck differences (compute-bound small batch, memory-bound large batch)
- **Avoid**: Just listing techniques without explaining *when* each matters or the latency/throughput tradeoff

## Answer
**Two goals:**
- **Latency**: Time to generate one response (matters for user experience)
- **Throughput**: Requests per second (matters for cost)

**Key techniques:**

**1. KV Caching** (covered separately)
- Reuse computed key-value pairs
- 2-10× speedup

**2. Batching**
- Process multiple requests in parallel
- Increases throughput, slight latency increase
- **Continuous batching**: Add/remove requests dynamically

**3. Model optimizations:**
- **Quantization**: 4-bit/8-bit weights (4× memory reduction)
- **Flash Attention**: Faster attention computation
- **GQA/MQA**: Smaller KV cache

**4. Speculative Decoding**
- Use small "draft" model to generate multiple tokens
- Large model verifies in parallel
- 2-3× speedup when draft model is good

**5. PagedAttention (vLLM)**
- Treat KV cache like virtual memory (pages)
- Reduce fragmentation, increase batch size
- 2× throughput improvement

**6. Tensor Parallelism**
- Split model across GPUs
- Lower latency (model too big for 1 GPU)

**7. Operator Fusion**
- Combine operations into single kernel
- Reduce memory I/O
- Examples: Flash Attention, fused layernorm

**Latency breakdown (typical):**
```
Prefill (process prompt): 10-50ms (depends on length)
Decode (per token): 20-100ms
Total: Prefill + (num_tokens * decode_time)
```

**Bottlenecks:**

- **Small batch**: Compute-bound (underutilizing GPU)
- **Large batch**: Memory-bound (KV cache access)

**Modern stacks:**
- **vLLM**: PagedAttention, continuous batching
- **TensorRT-LLM**: Operator fusion, quantization
- **llama.cpp**: CPU-optimized, quantization

**Throughput vs latency trade-off:**
- Large batch → high throughput, high latency
- Small batch → low latency, low throughput

## Follow-up Questions
- What's the bottleneck for single-request inference?
- How does speculative decoding work?
- What's continuous batching?
