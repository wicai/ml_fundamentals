# Temperature Effect

**Category:** modern_llm
**Difficulty:** 2
**Tags:** sampling, generation

## Question
What happens when you increase temperature in sampling from an LLM?

## Answer
**Higher temperature (T > 1)**: More random, flatter probability distribution
**Lower temperature (T < 1)**: More deterministic, sharper distribution
**T = 0**: Greedy decoding (always most likely token)

Formula: p_i = softmax(logits / T)_i

Practical: T=0.7 for chat, T=0.0 for factual QA, T=1.0 for creative writing.
