# Chain-of-Thought Prompting

**Category:** evaluation
**Difficulty:** 3
**Tags:** prompting, reasoning, emergence

## Question
What is chain-of-thought prompting and why does it improve reasoning?

## Answer
**Chain-of-Thought (CoT)**: Prompt model to show intermediate reasoning steps before final answer.

**Standard prompting:**
```
Q: Roger has 5 balls. He buys 2 cans of 3 balls. How many balls does he have?
A: 11
```

**CoT prompting:**
```
Q: Roger has 5 balls. He buys 2 cans of 3 balls. How many balls does he have?
A: Roger started with 5 balls. 2 cans of 3 balls each is 6 balls.
   5 + 6 = 11. The answer is 11.
```

**Why it works:**

1. **Computation space**: More tokens = more compute = harder problems solvable
2. **Explicit reasoning**: Forces model to break down problem
3. **Error correction**: Model can catch its own mistakes mid-generation
4. **Interpretability**: Humans can see reasoning process

**Emergence**: Only helps for large models (>10B params). Hurts performance for small models!

**Variants:**

**Zero-Shot CoT:**
```
Q: [problem]
A: Let's think step by step.
```
Surprisingly effective with just this prefix!

**Self-Consistency:**
- Generate multiple CoT paths
- Take majority vote of answers
- Improves accuracy significantly

**Least-to-Most:**
- Break problem into subproblems
- Solve each sequentially

**Performance gains:**
- Math: 50% → 80% accuracy (GSM8K)
- Reasoning: 30% → 60% accuracy (various benchmarks)

**Gotchas:**

1. **Hallucinated reasoning**: Steps might sound good but be wrong
2. **Unfaithful**: Reasoning might be post-hoc rationalization
3. **Longer = costlier**: More tokens = more inference cost
4. **Model size**: Doesn't work well for models <10B parameters

**Modern applications:**
- ChatGPT o1 (rumored to use CoT internally)
- Math/reasoning benchmarks
- Code generation (explain before writing)

## Follow-up Questions
- Why does CoT fail for small models?
- What's the difference between CoT and scratchpad?
- How do you evaluate CoT quality?
