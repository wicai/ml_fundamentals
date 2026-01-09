# Beam Search

**Category:** modern_llm
**Difficulty:** 3
**Tags:** inference, generation, decoding

## Question
What is beam search and when is it better than sampling?

## What to Cover
- **Set context by**: Explaining beam search as exploring multiple high-probability paths
- **Must mention**: The algorithm (keep top-k sequences), comparison with greedy and sampling, length normalization, when to use (translation, summarization) vs not (chat, creative)
- **Show depth by**: Discussing problems (generic outputs, no diversity) and the modern trend (sampling preferred for LLMs)
- **Avoid**: Only describing the algorithm without explaining when it's appropriate vs inappropriate

## Answer
**Beam Search**: Keep top-k most likely sequences at each step.

**Algorithm:**
```
beam_width = k (e.g., 5)

Step 1: Generate first token
  Keep top-k tokens → k sequences

Step 2: For each of k sequences
  Generate next token (vocabulary_size options)
  Total: k × vocabulary_size candidates
  Keep top-k overall → k sequences

Repeat until all sequences end or max_length
Return highest-scoring sequence
```

**Example (k=2):**
```
Start: ""

Step 1:
  Candidates: "The"(0.5), "A"(0.3), "It"(0.2), ...
  Beam: ["The", "A"]

Step 2 from "The":
  "The cat"(0.3), "The dog"(0.2), ...
Step 2 from "A":
  "A dog"(0.2), "A cat"(0.15), ...

  Top-2 overall: ["The cat", "The dog"]

Continue...
```

**Scoring:**
```
Score = log P(y_1, ..., y_n)
      = Σ log P(y_i | y_<i)

Often add length normalization:
Score = (1/n) Σ log P(y_i | y_<i)
```

**vs Greedy:**
- Greedy: Locally optimal at each step
- Beam: Explores multiple paths, more globally optimal

**vs Sampling:**
- Beam: Deterministic, finds high-probability outputs
- Sampling: Stochastic, more diverse outputs

**When to use beam search:**

✓ Machine translation (want most likely translation)
✓ Summarization (want faithful summary)
✓ Tasks where there's a "correct" output

✗ Creative generation (too boring, repetitive)
✗ Chatbots (not human-like)
✗ Code generation (sampling works better empirically)

**Problems with beam search:**

1. **Generic outputs**: "The dog is a dog" (high probability, boring)
2. **No diversity**: All beams converge to similar outputs
3. **Poor long-form**: Degrades for very long sequences

**Modifications:**

- **Diverse beam search**: Penalize similar beams
- **Constrained beam search**: Force certain tokens to appear
- **Length penalty**: Prefer longer/shorter sequences

**Modern trend for LLMs**:
Sampling (temperature + top-p) is standard, beam search less common.
Translation still uses beam search.

## Follow-up Questions
- Why does beam search produce boring outputs for chatbots?
- What's the computational cost of beam search?
- How do you choose beam width?
