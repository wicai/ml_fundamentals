# KV Cache Purpose

**Category:** modern_llm
**Difficulty:** 2
**Tags:** inference, optimization

## Question
What is KV caching and what problem does it solve?

## Answer
**KV caching** stores the Key and Value matrices from previous tokens during autoregressive generation.

**Problem solved**: Without caching, must recompute attention for all previous tokens at each step (wasteful). With caching, only compute new token and reuse cached K,V → ~2x speedup.
