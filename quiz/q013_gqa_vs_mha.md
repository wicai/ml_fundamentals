# GQA vs MHA

**Category:** modern_llm
**Difficulty:** 3
**Tags:** attention, architecture

## Question
How does Grouped-Query Attention (GQA) reduce KV cache size?

## Answer
GQA shares K and V across groups of Q heads instead of having separate K,V for each head.

Example: 32 Q heads, 8 KV heads → 4 Q heads share each KV head
→ KV cache is 4× smaller than standard Multi-Head Attention.
