# Continual Learning & Catastrophic Forgetting

**Category:** training
**Difficulty:** 3
**Tags:** training, finetuning, catastrophic_forgetting

## Question
What is catastrophic forgetting and how do you mitigate it?

## What to Cover
- **Set context by**: Explaining what catastrophic forgetting is (fine-tuning overwrites old knowledge)
- **Must mention**: Mitigation strategies (rehearsal/replay, EWC regularization, adapters/LoRA, distillation), their tradeoffs
- **Show depth by**: Discussing LLM-specific practices (mixing tasks during instruction tuning, continual pretraining with replay)
- **Avoid**: Only describing the problem without explaining practical solutions

## Answer
**Catastrophic Forgetting**: When fine-tuning on new task, model forgets old tasks.

**Example:**
```
1. Pretrain on general text
2. Fine-tune on medical text
3. Model now poor at general text!

Medical accuracy: 90% → 95% ✓
General accuracy: 85% → 60% ✗
```

**Why it happens:**

- Gradient updates overwrite old knowledge
- New task's weights incompatible with old task's weights
- Neural networks don't naturally separate task knowledge

**Mitigation strategies:**

**1. Rehearsal / Experience Replay**
```
Mix old task data with new task data during fine-tuning

Fine-tune batch:
  80% new task examples
  20% old task examples (sampled)

Prevents forgetting by continually practicing old tasks
```

**2. Regularization (EWC - Elastic Weight Consolidation)**
```
Identify important weights for old tasks (using Fisher information)
Penalize changing those weights during new task training

Loss = Task_loss + λ * Σ F_i * (θ_i - θ*_i)²

where:
  F_i = importance of weight i for old task
  θ*_i = weight value after old task
```

**3. Parameter isolation**

**Adapter layers:**
- Freeze base model
- Add small trainable adapters per task
- Each task has separate adapters

**LoRA:**
- Add task-specific low-rank matrices
- Base model frozen
- Swap adapters at inference

**4. Knowledge distillation**
```
Keep old model as teacher
Train new model to match:
  1. New task performance
  2. Old model's outputs (on old tasks)

Preserves old behavior while learning new
```

**5. Progressive networks**
```
Add new columns/modules for new tasks
Old columns frozen
New columns can use old columns (lateral connections)

Never forget, but model grows with each task
```

**For LLMs specifically:**

**Instruction tuning approach:**
```
Mix all tasks in one dataset:
  - Math problems
  - Coding
  - General QA
  - Domain-specific

Fine-tune on mixture
Prevents forgetting by practicing all tasks
```

**Continual pretraining:**
```
Want to update knowledge (e.g., new events, facts)
Challenge: Don't forget old facts

Solution:
  - Mix old and new data (replay)
  - Low learning rate
  - Gradient regularization
```

**Examples:**

**ChatGPT:**
- Pretrain → SFT → RLHF
- Each stage includes data from previous stages
- Prevents catastrophic forgetting

**Domain adaptation:**
```
Medical LLM from general LLM:
  - Mix 80% medical + 20% general during fine-tuning
  - Maintains general capability
```

**Trade-offs:**

| Method | Forgetting | Compute | Memory |
|--------|------------|---------|--------|
| Rehearsal | Low | Medium | High (store old data) |
| EWC | Medium | Low | Low |
| Adapters | None | Low | Medium (multiple adapters) |
| Progressive | None | Medium | High (grows) |

**Modern practice:**
- Rehearsal (mixing data) is most common
- LoRA/adapters for efficient task-specific models
- Full retraining on mixture for production systems

## Follow-up Questions
- Why is catastrophic forgetting worse for small fine-tuning datasets?
- How does rehearsal compare to EWC?
- Can you have zero forgetting?
