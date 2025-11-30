# Scaling Laws for LLMs

**Category:** modern_llm
**Difficulty:** 4
**Tags:** scaling, training, performance

## Question
What are the key findings from scaling laws research? How should you allocate compute budget?

## Answer
**Key Finding (Kaplan et al. 2020, Chinchilla 2022):**
Loss scales as a power law with model size (N), data size (D), and compute (C).

**Chinchilla Scaling Laws (Optimal):**
For compute budget C:
- Model parameters N ∝ C^0.5
- Training tokens D ∝ C^0.5

**Translation**: Double compute → √2 larger model AND √2 more data

**Old approach (GPT-3):** Large model, undertrained
- GPT-3: 175B params, 300B tokens
- Chinchilla optimal: 70B params, 1.4T tokens (same compute, better loss)

**Practical implications:**

1. **Most models are undertrained**: Need way more data than people thought
2. **Bigger isn't always better**: 70B model with more data beats 175B with less
3. **Data is crucial**: Data quality and quantity matter as much as parameters

**Emergence:** Some capabilities only appear at scale (few-shot learning, chain-of-thought). No known threshold, just appears suddenly.

**Modern trend (2024):**
- LLaMA 2: 70B params, 2T tokens (Chinchilla-optimal-ish)
- Smaller models trained longer: Phi-3, Mistral
- Focus shifting to data quality, not just quantity

**Gotcha**: Scaling laws assume good data. Garbage data doesn't follow power law.

## Follow-up Questions
- What is the "emergent abilities" debate?
- How do scaling laws change for downstream tasks vs pretraining loss?
- What's the practical limit of scaling?
