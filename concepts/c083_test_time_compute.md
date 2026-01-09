# Test-Time Compute Scaling

**Category:** inference
**Difficulty:** 4
**Tags:** scaling, inference, reasoning

## Question
What is test-time compute scaling and why is it important for reasoning models?

## What to Cover
- **Set context by**: Explaining the paradigm shift (more inference compute → better answers, not just training compute)
- **Must mention**: Techniques (longer CoT, best-of-N sampling, beam search, self-consistency, iterative refinement), when it helps most (high-variance reasoning tasks)
- **Show depth by**: Discussing scaling laws (log-linear improvements), cost tradeoffs, and when NOT to use (factual QA, simple tasks)
- **Avoid**: Only describing techniques without explaining the paradigm shift or when test-time compute actually helps

## Answer
**Test-time compute**: Using more computation during inference (not training) to improve answer quality.

**Traditional paradigm:**
```
More training compute → Better model → Better answers
Inference compute is fixed (one forward pass)
```

**New paradigm:**
```
More inference compute → Better answers (for same model)
Can trade latency for quality on a per-query basis
```

**Why this matters:**

**Old way: Scale training**
```
GPT-3:   $10M training  → 70% accuracy
GPT-4:   $100M training → 80% accuracy

Problem: Can't improve answers after training
```

**New way: Scale test-time**
```
o1 model trained once →
  Fast mode (1s):  70% accuracy
  Think mode (60s): 85% accuracy

Problem-specific: Easy question → fast, hard question → slow
```

**Techniques for test-time scaling:**

**1. Longer chain-of-thought**
```
Q: [Hard math problem]

Regular: Generate 100 tokens of reasoning
Extended: Generate 10,000 tokens of reasoning

More tokens → more "thinking space" → better answers
```

**2. Best-of-N sampling**
```
Generate N solutions (e.g., N=64)
Score each solution (with verifier model)
Return highest-scoring answer

Cost: N× slower
Benefit: Can find rare correct solutions
```

**3. Beam search over reasoning**
```
Standard: Sample one reasoning path
Beam search: Explore k=10 paths in parallel
Prune bad paths, expand promising ones

Like tree search through solution space
```

**4. Self-consistency**
```
Generate 20 different reasoning paths
Take majority vote of final answers

Example:
  Path 1→5: Answer = 42  (5 votes)
  Path 6→20: Answer = 43 (15 votes)
  Return: 43 ✓
```

**5. Iterative refinement**
```
1. Generate initial answer
2. Critique it (find errors)
3. Revise based on critique
4. Repeat 2-3 until confident

Like self-editing a paper
```

**Example: Coding problem**

```
Problem: Implement quicksort

1s (baseline):
  Generate code directly → 60% pass rate

10s (chain-of-thought):
  Plan algorithm → Write code → 75% pass rate

60s (search + verify):
  Generate 5 solutions
  Test each on examples
  Pick best → 90% pass rate
```

**Scaling laws for test-time compute:**

Research shows **log-linear improvements**:
```
2× compute → +5% accuracy
10× compute → +10% accuracy
100× compute → +15% accuracy

Diminishing returns, but still helps!
```

**When test-time compute helps most:**

**High variance tasks (helps a lot):**
- Math proofs
- Competitive programming
- Scientific reasoning
- Creative problem solving

**Low variance tasks (helps less):**
- Factual QA ("What is Paris?")
- Classification ("Sentiment: positive/negative")
- Simple lookup

**Why? Thinking longer doesn't give you new facts!**

**Cost-performance tradeoff:**

```
Model: o1-like reasoning model

Easy question: "What is 2+2?"
  Optimal: 1s, $0.01

Hard question: "Prove Fermat's Last Theorem for n=3"
  Optimal: 60s, $1.00

Adaptive compute: Spend more where it helps!
```

**Implementation challenges:**

1. **When to stop thinking?**
   - Fixed budget? (always think 60s)
   - Confidence-based? (stop when certain)
   - Verifier-based? (stop when verifier says good)

2. **How to search reasoning space?**
   - Random sampling? (diverse but wasteful)
   - Beam search? (focused but can miss solutions)
   - MCTS? (good but slow)

3. **How to score partial reasoning?**
   - Need good verifier/reward model
   - Hard for open-ended problems

**Comparison to training-time compute:**

| Aspect | Training Compute | Test Compute |
|--------|------------------|--------------|
| When | Before deployment | Per query |
| Cost | One-time, huge | Per-query, variable |
| Benefit | All queries | Specific query |
| Scaling | Diminishing | Diminishing |
| Flexibility | Fixed after | Adjustable |

**The big insight:**

> "For reasoning tasks, inference-time search can match or exceed improvements from scaling model size."

Instead of:
- Training 10× bigger model → 2× cost forever

Do:
- Use 10× more test-time compute → 2× cost only for hard problems

**Future implications:**
- Models may stay same size, but get "smarter" via better search
- Users pay per thinking time, not per token
- Adaptive systems: easy=fast, hard=slow

## Follow-up Questions
- How do you decide when to use more test-time compute?
- What's the theoretical limit of test-time compute scaling?
- How does this relate to human problem-solving?
