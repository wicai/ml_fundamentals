# Direct Preference Optimization (DPO)

**Category:** rlhf
**Difficulty:** 4
**Tags:** rlhf, dpo, alignment

## Question
What is DPO and how does it differ from PPO-based RLHF?

## What to Cover
- **Set context by**: Explaining the complexity/instability of PPO-based RLHF that DPO solves
- **Must mention**: The key insight (reparameterize RM in terms of optimal policy), the loss function intuition, advantages (simplicity, stability, no RM needed)
- **Show depth by**: Discussing tradeoffs (less flexible, tied to specific dataset) and adoption (LLaMA 3, Mistral)
- **Avoid**: Skipping the comparison with PPO pipeline or not explaining *why* DPO works

## Answer
**DPO**: Directly optimize policy on preference data without training separate reward model or using RL.

**Key insight**:
Reparameterize reward model in terms of optimal policy, then optimize policy directly.

**PPO-RLHF pipeline:**
1. Train reward model on preferences
2. Use RL (PPO) to optimize policy against RM
→ Complex, unstable, requires RL engineering

**DPO pipeline:**
1. Directly train policy on preference pairs
→ Simple supervised learning!

**DPO loss:**
```
L = -log σ(β * log(π_θ(y_w|x) / π_ref(y_w|x))
           - β * log(π_θ(y_l|x) / π_ref(y_l|x)))

where:
  y_w = preferred completion
  y_l = dispreferred completion
  π_ref = reference policy (SFT model, frozen)
  β = KL penalty strength
```

**Intuition**: Increase probability of preferred outputs relative to reference, decrease probability of dispreferred outputs.

**Advantages over PPO:**

1. **Simplicity**: No RM training, no RL loop, just supervised learning
2. **Stability**: No reward hacking, no policy collapse
3. **Efficiency**: Single training phase vs 2 (RM + PPO)
4. **Performance**: Often matches or beats PPO

**Disadvantages:**

1. **Less flexible**: Can't use explicit reward function
2. **Data efficiency**: Might need more preference pairs
3. **Generalization**: Tied to specific preference dataset

**When to use:**

- **DPO**: Simpler, good default choice
- **PPO**: Need explicit reward modeling, iterative data collection, complex reward functions

**Modern trend**: Many labs switching from PPO to DPO (Llama 3, Mistral).

## Follow-up Questions
- How does DPO implicitly model a reward function?
- Can you combine DPO with online data collection?
- What's the role of β in DPO?
