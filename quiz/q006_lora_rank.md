# LoRA Rank

**Category:** training
**Difficulty:** 2
**Tags:** lora, finetuning

## Question
What is a typical rank (r) value for LoRA and what does it control?

## Answer
**Typical rank**: r = 4 to 8 (8 is common starting point as of 2025)

**Controls**: The capacity of the low-rank adaptation. Higher rank = more parameters = more expressive but also more memory/compute.

LoRA adds matrices B (d×r) and A (r×k), so total params = r*(d+k).

**Modern practice**: Start with r=8. Only increase if needed. Many tasks work well with r=4.
