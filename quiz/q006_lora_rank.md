# LoRA Rank

**Category:** training
**Difficulty:** 2
**Tags:** lora, finetuning

## Question
What is a typical rank (r) value for LoRA and what does it control?

## Answer
**Typical rank**: r = 8 to 16 (16 is common sweet spot)

**Controls**: The capacity of the low-rank adaptation. Higher rank = more parameters = more expressive but also more memory/compute.

LoRA adds matrices B (d×r) and A (r×k), so total params = r*(d+k).
