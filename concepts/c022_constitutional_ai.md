# Constitutional AI

**Category:** rlhf
**Difficulty:** 3
**Tags:** alignment, rlhf, safety

## Question
What is Constitutional AI and how does it differ from standard RLHF?

## What to Cover
- **Set context by**: Explaining the problems with standard RLHF for harmlessness (expensive, traumatizing, inconsistent)
- **Must mention**: The two phases (self-critique SL, RLAIF with AI preferences), the constitution concept, key benefits (scalability, consistency, transparency)
- **Show depth by**: Noting the hybrid approach (human feedback for helpfulness, AI feedback for harmlessness)
- **Avoid**: Conflating CAI with pure RLAIF—the constitution and self-critique phase are key

## Answer
**Constitutional AI (CAI)**: Anthropic's method for training harmless AI using AI feedback instead of just human feedback.

**Standard RLHF problem:**
- Need thousands of human labels for harmfulness
- Expensive, slow, traumatizing for labelers
- Inconsistent across labelers

**Constitutional AI approach:**

**Phase 1: Supervised Learning (Self-critique)**
```
1. Generate harmful response
2. Critique it based on "constitution" (principles)
3. Revise to be harmless
4. Train on revised responses
```

Example constitution principle:
"Choose the response that is least intended to build a relationship with the user"

**Phase 2: RL from AI Feedback (RLAIF)**
```
1. Generate pairs of responses
2. Use LLM to judge which better follows constitution
3. Train reward model on AI preferences (not human)
4. Standard RLHF/PPO using this RM
```

**Benefits:**

1. **Scalability**: AI can label thousands of examples cheaply
2. **Consistency**: Same LLM gives consistent judgments
3. **Transparency**: Principles are explicit, not implicit in human labels
4. **Safety**: No traumatizing content for human labelers

**Trade-offs:**

- AI judges might miss subtle harm
- Constitution authoring requires expertise
- Still needs some human oversight

**Results (Claude):**
- Harmlessness comparable to human RLHF
- More transparent/steerable
- Scales better

**Key insight**: Human feedback for helpfulness, AI feedback for harmlessness.

## Follow-up Questions
- How do you write a good constitution?
- Can AI feedback fully replace human feedback?
- What are failure modes of RLAIF?
