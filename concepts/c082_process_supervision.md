# Process Supervision vs Outcome Supervision

**Category:** training
**Difficulty:** 4
**Tags:** rlhf, reasoning, reward_modeling

## Question
What is the difference between process supervision and outcome supervision for training reasoning models?

## Answer
**Two approaches to reward modeling for multi-step reasoning:**

**Outcome Supervision (ORM - Outcome Reward Model):**
- Only reward final answer correctness
- Don't evaluate intermediate steps
- Simpler to implement, less data needed

**Process Supervision (PRM - Process Reward Model):**
- Reward each reasoning step independently
- Label individual steps as correct/incorrect
- More expensive but better performance

**Example:**

```
Problem: What is 15 × 24?

Reasoning trace:
Step 1: "Let me break this down: 15 × 24 = 15 × (20 + 4)"     [correct ✓]
Step 2: "15 × 20 = 300"                                        [correct ✓]
Step 3: "15 × 4 = 50"                                          [incorrect ✗]
Step 4: "300 + 50 = 350"                                       [correct logic ✓]
Step 5: "Answer: 350"                                          [wrong final answer ✗]

Outcome Supervision:
  Reward = 0 (final answer wrong)
  Model learns: "This entire approach is bad"

Process Supervision:
  Step 1: +1 (good decomposition)
  Step 2: +1 (correct)
  Step 3: -1 (15×4=60, not 50)
  Step 4: +1 (arithmetic correct given inputs)
  Step 5: -1 (wrong answer)

  Model learns: "Step 3 is where I made the mistake"
```

**Why process supervision works better:**

1. **Credit assignment**: Identifies exactly where reasoning failed
2. **Partial credit**: Rewards correct steps even if final answer wrong
3. **Generalization**: Learns systematic reasoning, not just answer patterns
4. **Reduces reward hacking**: Harder to game with lucky guesses

**Data requirements:**

**Outcome supervision:**
```
Problem: [math problem]
Answer: [correct answer]
Model solution: [reasoning + answer]
Label: ✓ or ✗

~1000s of examples needed
```

**Process supervision:**
```
Problem: [math problem]
Step 1: [reasoning]    Label: ✓/✗
Step 2: [reasoning]    Label: ✓/✗
...
Step n: [answer]       Label: ✓/✗

~100,000s of step labels needed
```

**OpenAI's findings (2023):**
- Tested on math problems (MATH dataset)
- Process-supervised model: **78%** accuracy
- Outcome-supervised model: **72%** accuracy
- Human baseline: **~90%**

**Key result: Process supervision reduces hallucinated reasoning**

**Implementation approaches:**

**1. Human labeling (gold standard, expensive)**
```
Show humans each reasoning step
Label as correct/incorrect/neutral
Aggregate labels
```

**2. Automated verification (when possible)**
```
For math: Check if step follows from previous
For code: Run tests on intermediate outputs
For logic: Formal verification
```

**3. Model-based labeling (cheaper, less reliable)**
```
Use stronger model to label steps
Train weaker model on these labels
"Distillation" of reasoning ability
```

**4. Self-consistency (no labels needed)**
```
Generate multiple reasoning paths
Steps that appear in correct paths → positive
Steps that appear in wrong paths → negative
```

**Challenges:**

1. **Ambiguous steps**: "Let me think about this" - correct or not?
2. **Alternative paths**: Correct answer via wrong reasoning
3. **Partial credit**: How much reward for partially correct step?
4. **Scalability**: Need many human labels

**Connection to o1:**
- Likely uses process supervision heavily
- Possibly combined with outcome supervision
- May use model-generated process labels (cheaper to scale)

**Best practice: Combine both!**
```
Primary: Process supervision (guides reasoning)
Auxiliary: Outcome supervision (ensures correctness)

Loss = α * L_process + β * L_outcome
```

**Analogy: Teaching a student**
```
Outcome only: "Your answer is wrong, try again"
Process: "Step 3 is wrong because you forgot to carry the 1.
          Everything else looks good!"

Process supervision is better teaching!
```

## Follow-up Questions
- How do you label process steps at scale?
- Can you do process supervision without human labels?
- Why doesn't process supervision help for factual QA?
