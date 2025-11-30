# Reward Models in RLHF

**Category:** rlhf
**Difficulty:** 4
**Tags:** rlhf, reward, training

## Question
How do you train a reward model for RLHF? What architecture is used?

## Answer
**Reward Model (RM)**: Predicts scalar score representing human preference.

**Architecture:**
```
Input: prompt + completion
Transformer (pretrained LLM with same architecture as policy)
→ Pool last hidden state
→ Linear layer → scalar reward
```

**Training data:**
```
Prompt: "Explain quantum computing"
Completion A: [detailed, accurate explanation]
Completion B: [confusing, incorrect explanation]
Human preference: A > B
```

**Loss function (Bradley-Terry model):**
```
P(A > B) = σ(r_A - r_B)  # Sigmoid of reward difference

Loss = -log P(y_winner > y_loser)
     = -log σ(r_winner - r_loser)
```

**Training process:**

1. Start with SFT model checkpoint
2. Replace LM head with value head (scalar output)
3. Collect comparisons: sample pairs from policy, humans rank
4. Train to predict preferences

**Data collection:**

- Typical dataset: 50K-500K comparisons
- Each comparison might cost $1-5 (human labeling)
- Use the policy model to generate candidates (not just SFT model)

**Challenges:**

1. **Distribution shift**: RM trained on current policy outputs, but policy will drift during RL
2. **Overfitting**: Easy to overfit on comparison data
3. **Inconsistency**: Humans disagree (~30% of comparisons have disagreement)
4. **Gaming**: Model learns to exploit RM weaknesses (reward hacking)

**Solutions:**

- Ensemble RMs (train multiple, use average/median)
- Regularization (dropout, weight decay)
- Iterative collection (collect more data as policy improves)
- Uncertainty estimation (flag low-confidence predictions)

## Follow-up Questions
- Why use the LM architecture for reward model instead of a smaller classifier?
- How do you handle disagreement in human preferences?
- What is reward hacking?
