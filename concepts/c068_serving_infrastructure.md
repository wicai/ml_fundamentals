# LLM Serving Infrastructure

**Category:** modern_llm
**Difficulty:** 3
**Tags:** deployment, infrastructure, serving

## Question
What are the key components and challenges of serving LLMs in production?

## What to Cover
- **Set context by**: Framing serving as achieving low latency and high throughput at scale
- **Must mention**: Key components (batching, KV cache management, load balancing, autoscaling), serving frameworks (vLLM, TensorRT-LLM, TGI), main challenges (latency, cost, memory, cold start)
- **Show depth by**: Discussing architecture patterns (replicas, disaggregated prefill/decode) and best practices
- **Avoid**: Only listing components without explaining the challenges and tradeoffs (latency vs throughput)

## Answer
**Serving LLM** = Make model available for inference at scale with low latency and high throughput.

**Key components:**

**1. Model loading & initialization:**
```
Challenge: 70B model = 140GB in fp16
Solutions:
  - Model parallelism (split across GPUs)
  - Quantization (load in 4-bit = 35GB)
  - Lazy loading (load layers as needed)
```

**2. Request batching:**
```
Naive: Process one request at a time (underutilizes GPU)

Batching: Process N requests together
  - Higher throughput
  - Slightly higher latency per request

Challenge: Variable-length sequences
  - Padding wasteful
  - Solution: Continuous batching (vLLM)
```

**3. KV cache management:**
```
Problem: KV cache per request
  70B model, 2K tokens, batch=32: ~40GB just for KV cache

Solutions:
  - PagedAttention (vLLM): Virtual memory for KV cache
  - Preemption: Swap out idle requests
  - Eviction: Remove old requests
```

**4. Load balancing:**
```
Multiple model replicas
Distribute requests across replicas

Strategies:
  - Round-robin
  - Least loaded
  - Latency-based

Tools: Kubernetes, Ray Serve, Nginx
```

**5. Autoscaling:**
```
Scale up during high load, down during low load

Metrics:
  - Queue length
  - GPU utilization
  - Request latency

Challenge: Slow cold start (loading 140GB model)
```

**6. Monitoring & observability:**
```
Metrics to track:
  - Throughput (requests/sec)
  - Latency (p50, p95, p99)
  - GPU utilization
  - Memory usage
  - Error rate
  - Queue depth

Tools: Prometheus, Grafana, Datadog
```

**7. Caching:**
```
Cache common prompts/completions
  - Exact match: Hash-based
  - Semantic match: Embedding similarity

Prefix caching: Cache prompt encoding
  "Translate to French: [X]" → cache "Translate to French:"
```

**Serving frameworks:**

**vLLM:**
- PagedAttention for KV cache
- Continuous batching
- 2-4× higher throughput than naive

**TensorRT-LLM (Nvidia):**
- Optimized kernels (Flash Attention, fused ops)
- Quantization support (INT8, FP8)
- Multi-GPU inference

**Text Generation Inference (HuggingFace):**
- Production-ready
- Flash Attention, quantization
- Streaming support

**Ray Serve / Anyscale:**
- Distributed serving
- Autoscaling
- Multi-model serving

**llama.cpp:**
- CPU inference
- Quantization (GGUF format)
- MacOS Metal acceleration

**Challenges:**

**1. Latency:**
```
Target: <1s for interactive chat

Bottlenecks:
  - Model size (70B slower than 7B)
  - Sequence length (2K vs 8K)
  - Batch size (higher = slower per request)

Solutions:
  - Speculative decoding
  - Quantization
  - Faster hardware (H100 vs A100)
```

**2. Cost:**
```
H100 GPU: ~$3/hour
70B model needs 4-8× H100s
= $24/hour minimum

At scale: $1M+/month

Solutions:
  - Batching (amortize cost)
  - Quantization (fewer GPUs)
  - Cheaper GPUs (A100, L40S)
```

**3. Memory:**
```
70B model (fp16): 140GB
KV cache (batch=32, seq=2K): 40GB
Total: 180GB per replica

Exceeds single GPU (80GB H100)
→ Must use tensor parallelism (4× GPUs)
```

**4. Cold start:**
```
Loading 140GB model: 30-60 seconds

Not acceptable for autoscaling
Solution: Keep warm instances
```

**5. Fairness:**
```
Long requests block short requests

Solution: Preemption, fair queueing
```

**Architecture patterns:**

**1. Replicas:**
```
N identical model copies
Load balancer distributes requests

Simple, scales linearly
```

**2. Disaggregated prefill/decode:**
```
Prefill server: Process prompts (compute-bound)
Decode server: Generate tokens (memory-bound)

Optimize each separately
```

**3. Model variants:**
```
Small model (7B): Fast, cheap, most requests
Large model (70B): Slow, expensive, hard requests

Route based on complexity
```

**Best practices:**

1. **Use serving framework** (vLLM, TRT-LLM), don't roll your own
2. **Monitor everything**: Latency, throughput, GPU util
3. **Batch aggressively**: Higher throughput
4. **Cache prompts**: Prefix caching for common patterns
5. **Quantize**: 4-bit/8-bit for cheaper serving

## Follow-up Questions
- What is continuous batching?
- How does PagedAttention improve serving?
- What's the trade-off between latency and throughput?
