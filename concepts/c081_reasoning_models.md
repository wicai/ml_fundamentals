# Reasoning Models (o1-style)

**Category:** modern_llm
**Difficulty:** 4
**Tags:** reasoning, inference, o1

## Question
What are reasoning models like o1 and how do they differ from standard LLMs?

## Answer
**Reasoning models**: LLMs trained to think longer at inference time, generating internal reasoning traces before answering.

**Key difference from standard LLMs:**
```
Standard LLM (GPT-4):
Question → [Single Forward Pass] → Answer
Fast, fixed compute

Reasoning Model (o1):
Question → [Extended Chain of Thought] → Answer
Slow, variable compute (seconds to minutes)
```

**How o1-style models work:**

1. **Extended thinking time**
   - Generate long internal monologue (not shown to user)
   - Can be thousands of tokens of reasoning
   - Model explores multiple approaches, backtracks, self-corrects

2. **Process supervision during training**
   - Reward correct reasoning steps, not just final answer
   - Learn to verify own work
   - Develop systematic problem-solving strategies

3. **Test-time compute scaling**
   - More thinking time → better performance
   - Unlike standard models where more tokens ≠ better answers
   - Can allocate compute based on problem difficulty

**Example reasoning trace (simplified):**
```
Q: Prove that sqrt(2) is irrational.

Internal reasoning (hidden from user):
- Let me try proof by contradiction
- Assume sqrt(2) = p/q where p,q are coprime integers
- Then 2 = p²/q², so p² = 2q²
- This means p² is even, so p is even
- Let p = 2k, then (2k)² = 2q²
- So 4k² = 2q², which means 2k² = q²
- Wait, this means q is also even
- But we assumed p,q are coprime!
- Contradiction found ✓

User-facing answer:
We can prove this by contradiction...
```

**Training approach (speculated for o1):**

1. **RL with process supervision**
   - Reward each reasoning step, not just outcome
   - Use outcome-based reward modeling (ORM) + process-based reward modeling (PRM)
   - Encourage exploration of solution space

2. **Self-verification**
   - Train model to check its own work
   - Generate solution, then critique it
   - Iterate until confident

3. **Search at inference**
   - Potentially use beam search or MCTS over reasoning paths
   - Generate multiple reasoning traces, pick best

**Performance characteristics:**

**Strengths:**
- Math: ~84% on AIME (vs ~13% for GPT-4)
- Coding: Top 89th percentile on Codeforces
- Science: PhD-level physics/chemistry problems
- Planning: Multi-step reasoning tasks

**Weaknesses:**
- Slow: 10-60s per response vs <1s for GPT-4
- Expensive: More tokens = higher cost
- Factual recall: No better than GPT-4 (reasoning ≠ knowledge)
- Simple questions: Overkill, still uses full reasoning budget

**Key insight: Test-time compute matters!**
```
Standard scaling: More training compute → better model
New paradigm: More inference compute → better answers

Implication: Can trade latency for quality per query
```

**Comparison:**

| Aspect | Standard LLM | Reasoning Model |
|--------|-------------|-----------------|
| Speed | Fast (<1s) | Slow (10-60s) |
| Compute | Fixed per token | Variable per problem |
| Reasoning | Implicit | Explicit internal |
| Math/Coding | Good | Excellent |
| General QA | Excellent | Excellent |
| Cost | Low | High |

**Open questions:**
- How much of the improvement is from RL vs longer thinking?
- Is the reasoning faithful or post-hoc?
- Can we get similar gains more efficiently?

## Follow-up Questions
- What is process supervision vs outcome supervision?
- How does test-time compute scaling work?
- Why can't we just prompt GPT-4 to think longer?
