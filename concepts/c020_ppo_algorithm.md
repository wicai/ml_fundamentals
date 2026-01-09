# PPO for RLHF

**Category:** rlhf
**Difficulty:** 5
**Tags:** rlhf, ppo, reinforcement_learning

## Question
How does PPO (Proximal Policy Optimization) work in the context of RLHF?

## What to Cover
- **Set context by**: Explaining PPO's role as the RL algorithm that optimizes against the reward model
- **Must mention**: The objective (reward - KL penalty), why KL penalty is critical (stability, capability preservation), clipping mechanism (ε=0.2)
- **Show depth by**: Mentioning RLHF-specific challenges (large action space, sparse rewards) and the DPO alternative
- **Avoid**: Getting lost in generic PPO details without connecting to the RLHF context

## Answer
**PPO**: RL algorithm that optimizes policy while preventing too-large updates.

**Objective:**
```
Maximize: E[reward(x, y)] - β * KL(π || π_SFT)

where:
  reward(x, y) = RM(x, y)  # Reward model score
  KL term = prevents drifting too far from SFT model
  β ∈ [0.01, 0.1] controls strength
```

**Why KL penalty:**
1. **Stability**: Prevents policy collapse (gibberish that fools RM)
2. **Capability preservation**: Keeps language modeling ability from pretraining
3. **Distribution**: Keeps outputs in-distribution for RM

**PPO algorithm (simplified):**

```
1. Sample batch of prompts x
2. Generate completions y ~ π_θ(y|x)
3. Compute rewards r = RM(x, y) - β * log(π_θ(y|x) / π_SFT(y|x))
4. Compute advantages A (how much better than expected)
5. Update policy with clipped objective:

   L = min(
     r_t(θ) * A,  # Standard policy gradient
     clip(r_t(θ), 1-ε, 1+ε) * A  # Clipped version
   )

   where r_t(θ) = π_θ(y|x) / π_old(y|x)  # Importance ratio
```

**The clipping (ε=0.2):**
- Prevents too-large policy updates
- If ratio > 1+ε (policy increased prob too much) → clip
- If ratio < 1-ε (policy decreased prob too much) → clip

**RLHF-specific challenges:**

1. **Large action space**: Every token is an action, combinatorial explosion
2. **Sparse rewards**: Reward only at end of sequence
3. **Expensive rollouts**: Generating text is slow
4. **RM distribution shift**: RM hasn't seen new policy outputs

**Hyperparameters:**
- Learning rate: ~1e-6 (much smaller than pretraining)
- β (KL penalty): 0.01-0.1
- Clip ε: 0.2
- Episodes: 10K-100K

**Modern alternative: DPO** (Direct Preference Optimization)
- Skips RL entirely, directly optimizes policy on preferences
- Simpler, more stable, often better results

## Follow-up Questions
- Why is PPO preferred over other RL algorithms?
- What is reward hacking and how does KL penalty help?
- How is DPO different from PPO?
