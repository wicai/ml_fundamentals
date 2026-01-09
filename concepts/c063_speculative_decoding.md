# Speculative Decoding

**Category:** modern_llm
**Difficulty:** 4
**Tags:** inference, optimization, efficiency

## Question
What is speculative decoding and how does it speed up LLM inference?

## What to Cover
- **Set context by**: Explaining the sequential bottleneck in autoregressive generation
- **Must mention**: The algorithm (draft with small model, verify with large model in parallel), why it's mathematically exact, typical speedups (2-3×)
- **Show depth by**: Discussing draft model strategies (smaller version, distilled model) and variants (Medusa, lookahead decoding)
- **Avoid**: Skipping the explanation of why large model verification can be done in parallel (same cost as 1 token)

## Answer
**Problem**: Autoregressive generation is sequential - must generate one token at a time.

**Observation**: Many tokens are "easy" and predictable. Why use giant model for every token?

**Speculative Decoding**: Use small fast "draft" model to guess multiple tokens, verify with large model in parallel.

**Algorithm:**
```
1. Draft model generates k tokens speculatively:
   [t_1, t_2, ..., t_k] ~ small_model

2. Large model verifies all k tokens in parallel:
   Compute P_large(t_i | context) for all i

3. Accept or reject each token:
   For i = 1 to k:
     Sample u ~ Uniform(0,1)
     If u < P_large(t_i) / P_draft(t_i):
       Accept t_i
     Else:
       Reject t_i and all following
       Resample from P_large

4. Repeat from last accepted token
```

**Example:**
```
Draft model guesses (k=4):
  "The cat sat on"

Large model verifies:
  P_large("The") = 0.9, P_draft("The") = 0.8 → Accept
  P_large("cat") = 0.7, P_draft("cat") = 0.6 → Accept
  P_large("sat") = 0.3, P_draft("sat") = 0.8 → Reject!

Result: Accept 2 tokens instead of 1
  If draft was perfect, would accept all 4
```

**Why it works:**

1. **Parallelism**: Large model processes k tokens in one forward pass (same cost as 1 token)
2. **Good drafts**: If draft model agrees with large model, get k× speedup
3. **Exact sampling**: Mathematically equivalent to sampling from large model alone

**Speedup:**
```
k = 4 speculative tokens
Draft accuracy (acceptance rate) = 70%

Expected tokens per iteration:
  0.7^0 * 0 + 0.7^1 * 1 + 0.7^2 * 2 + 0.7^3 * 3 + 0.7^4 * 4
  ≈ 2.4 tokens per large model call

Speedup: 2.4× (compared to 1 token per call)
```

**Typical speedups: 2-3× for k=4-8**

**Draft model strategies:**

**1. Smaller version of same model:**
```
Large: 70B parameters
Draft: 7B parameters (10× smaller, 10× faster)

Naturally aligned (trained on same data/objective)
```

**2. Distilled model:**
```
Train small model to mimic large model's outputs
Better alignment → higher acceptance rate
```

**3. Specialized draft model:**
```
Train draft model to maximize acceptance rate
Not just next-token prediction
```

**4. Retrieval-augmented drafting:**
```
Draft based on similar retrieved examples
Non-parametric approach
```

**Trade-offs:**

✓ 2-3× faster inference
✓ Exact sampling (no quality loss)
✓ Compatible with KV cache
✗ Need small model (more memory)
✗ Draft model must be fast enough
✗ Less effective if draft/large models very different

**Variants:**

**Medusa:**
- Single model with multiple heads
- Each head predicts future token at different position
- Self-speculative decoding

**Lookahead decoding:**
- Use n-gram matching for drafting
- No draft model needed

**Assisted generation (HuggingFace):**
- Implementation of speculative decoding
- Easy to use with any model pair

**When effective:**

✓ Large model is expensive (70B+)
✓ Good small model available (7B)
✓ Non-creative generation (more predictable)
✗ Very creative/random generation
✗ Small models already fast

## Follow-up Questions
- Why is speculative decoding mathematically exact?
- How do you choose k (number of speculative tokens)?
- What makes a good draft model?
