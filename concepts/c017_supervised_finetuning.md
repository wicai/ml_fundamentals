# Supervised Fine-Tuning (SFT)

**Category:** rlhf
**Difficulty:** 3
**Tags:** finetuning, alignment, training

## Question
What is supervised fine-tuning and how does it differ from pretraining?

## What to Cover
- **Set context by**: Placing SFT in the training pipeline (pretrain → SFT → RLHF)
- **Must mention**: The data format (prompt, completion pairs), key differences from pretraining (data size, compute, objective), what SFT teaches (instruction following, format, style)
- **Show depth by**: Noting gotchas (data quality > quantity, can reduce capabilities if overfitted)
- **Avoid**: Conflating SFT with full alignment (SFT alone doesn't prevent harmful outputs)

## Answer
**Supervised Fine-Tuning (SFT)**: Continue training pretrained LLM on high-quality demonstrations of desired behavior.

**Data format:**
```
Prompt: "Explain photosynthesis to a 5-year-old"
Completion: "Plants are like little factories that make food from sunlight! ..."
```

**vs Pretraining:**

| Aspect | Pretraining | SFT |
|--------|-------------|-----|
| Data | Raw internet text (trillions of tokens) | Curated demos (10K-100K examples) |
| Objective | Predict next token (unsupervised) | Predict completion given prompt (supervised) |
| Goal | General language understanding | Follow instructions, desired format/style |
| Compute | Massive (10,000s of GPU-days) | Small (10-100 GPU-days) |

**SFT process:**

1. Start with pretrained model (e.g., GPT-3)
2. Create dataset of (prompt, good_completion) pairs
3. Train with standard next-token prediction loss
4. Only compute loss on completion tokens (mask prompt in loss)

**What SFT teaches:**

- **Instruction following**: Respond to "Write a poem" vs just continuing text
- **Format**: Structured outputs (lists, code, etc.)
- **Style**: Helpful, harmless, honest tone
- **Domain knowledge**: Specialized tasks (coding, math)

**Gotchas:**

- **Data quality matters more than quantity**: 10K great examples > 100K mediocre
- **Can reduce capabilities**: Over-fitting on narrow SFT data can hurt general ability
- **Not sufficient for alignment**: SFT alone doesn't prevent harmful outputs

**Modern pipeline**: Pretrain → SFT → RLHF (PPO/DPO)

## Follow-up Questions
- Why mask the prompt in the loss calculation?
- How much does SFT data size matter?
- Can SFT teach the model new knowledge?
