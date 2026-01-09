# RLHF Overview

**Category:** rlhf
**Difficulty:** 4
**Tags:** rlhf, alignment, training

## Question
What is RLHF (Reinforcement Learning from Human Feedback) and why is it used?

## What to Cover
- **Set context by**: Explaining what SFT alone can't solve (hard to specify "good" behavior, comparison easier than generation)
- **Must mention**: The three stages (SFT, Reward Model, RL/PPO), the KL penalty purpose, examples of models using RLHF
- **Show depth by**: Mentioning challenges (expensive, reward hacking, distribution shift) and alternatives (DPO)
- **Avoid**: Only describing the pipeline without explaining *why* comparison-based learning is better than direct supervision

## Answer
**RLHF**: Train a language model to maximize human preferences using RL.

**The problem SFT doesn't solve:**
- Hard to write down "good" behavior (safety, helpfulness, harmlessness)
- Easier to compare two outputs than generate perfect output
- Some behaviors are subjective (style, tone)

**RLHF pipeline (3 stages):**

**1. Supervised Fine-Tuning (SFT)**
- Start with pretrained model
- Fine-tune on demonstrations
- Creates initial policy π_SFT

**2. Reward Model (RM) Training**
- Show humans pairs of outputs for same prompt
- Collect preferences: A > B
- Train model to predict human preferences: RM(x, y) = scalar score
- Loss: Maximize P(y_winner > y_loser)

**3. RL Optimization (PPO)**
- Use RM as reward signal
- Optimize policy to maximize: reward(x, y) - β * KL(π || π_SFT)
- KL penalty prevents model from drifting too far from SFT

**Why it works:**

1. **Comparison is easier**: Humans better at ranking than writing perfect outputs
2. **Captures preferences**: Safety, helpfulness, style are easier to show than tell
3. **Optimization**: RL finds high-reward behaviors humans didn't explicitly demonstrate

**Examples:**
- ChatGPT: SFT + RLHF
- Claude: RLHF + Constitutional AI
- Llama 2: RLHF for chat variant

**Challenges:**
- **Expensive**: Requires lots of human feedback
- **Reward hacking**: Model finds high-reward shortcuts (verbosity, sycophancy)
- **Distribution shift**: RL can produce outputs reward model hasn't seen

## Follow-up Questions
- Why is the KL penalty necessary?
- What is reward hacking and how do you prevent it?
- Can you do RLHF without the RL part? (Yes → DPO)
