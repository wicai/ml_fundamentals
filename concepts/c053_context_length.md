# Context Length & Long-Context Models

**Category:** modern_llm
**Difficulty:** 3
**Tags:** architecture, attention, scaling

## Question
What limits context length in transformers and how are modern models extending it?

## Answer
**Context length**: Maximum number of tokens the model can process at once.

**Why it matters:**
- Books, long documents, codebases
- Multi-turn conversations
- Retrieval with many documents

**Limitations:**

**1. Computational: O(n²) attention**
```
Sequence length 2K: 4M attention scores
Sequence length 32K: 1B attention scores (256× more!)
```

**2. Memory: KV cache**
```
LLaMA-7B, 32K context:
  KV cache: ~16GB (just for caching!)
```

**3. Position encoding**
- Learned embeddings don't generalize beyond training length
- Sinusoidal/RoPE degrade at long distances

**4. Quality degradation**
- "Lost in the middle" problem
- Model ignores middle of long contexts

**Approaches to extend context:**

**1. Better position encodings**

**ALiBi (Attention with Linear Biases):**
```
Add static bias based on distance to attention scores
No learned positions, extrapolates well
Can train on 2K, inference on 16K+
```

**RoPE with scaling:**
- YaRN, CodeLLaMA extended context
- Modified frequency computation
- Train on 2K → inference on 100K

**2. Flash Attention**
- O(n²) complexity, but O(n) memory
- Enables 10× longer contexts in practice

**3. Sparse attention**
- Only attend to subset of tokens
- Longformer, BigBird, LongNet
- O(n) instead of O(n²)

**4. Retrieval instead of context**
- Don't put everything in context
- Use RAG to fetch relevant chunks
- Unlimited "effective" context

**5. Compression**
- AutoCompressor: Compress past context into summary tokens
- Memorizing Transformers: Cache past with retrieval

**Modern models:**

| Model | Base | Extended | Method |
|-------|------|----------|--------|
| GPT-3 | 2K | 4K | - |
| GPT-4 | 8K | 128K | Unknown |
| Claude 2 | - | 100K | Unknown |
| Claude 3 | - | 200K | Unknown |
| Llama 2 | 4K | - | RoPE |
| CodeLlama | 4K | 100K | RoPE scaling |
| Gemini 1.5 | - | 1M | Unknown |

**"Lost in the middle" problem:**
```
Long context: [Info A] [Filler...] [Info B] [Filler...] [Query]

Model attends well to:
  ✓ Start (Info A)
  ✓ End (near query)
  ✗ Middle (Info B often ignored)

Solution: Put important info at start/end
```

**Practical limits:**
- Most tasks don't need >32K tokens
- Cost increases quadratically
- RAG often better than giant context

## Follow-up Questions
- What is the "lost in the middle" problem?
- How does Flash Attention enable longer contexts?
- What's better: long context or RAG?
