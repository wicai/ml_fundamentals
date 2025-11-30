# GPT Architecture Type

**Category:** modern_llm
**Difficulty:** 2
**Tags:** architecture, gpt

## Question
What type of architecture does GPT use: encoder-only, decoder-only, or encoder-decoder?

## Answer
**Decoder-only**

GPT uses causal self-attention (can only attend to previous tokens) and is trained with next-token prediction. No encoder or cross-attention needed.
