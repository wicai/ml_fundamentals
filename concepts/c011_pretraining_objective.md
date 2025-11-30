# Pretraining Objective for LLMs

**Category:** training
**Difficulty:** 3
**Tags:** training, objective, loss

## Question
What is the pretraining objective for models like GPT? Why does next-token prediction work so well?

## Answer
**Objective: Autoregressive Language Modeling**
```
Maximize: log P(x_1, x_2, ..., x_n)
       = Σ log P(x_t | x_<t)  # Chain rule

Loss: Cross-entropy = -log P(x_t | x_<t)
```

**In practice:**
- Input: tokens [x_1, x_2, ..., x_n]
- Target: shifted [x_2, x_3, ..., x_{n+1}]
- Predict next token at every position in parallel

**Why does this work so well?**

1. **Self-supervised**: No labels needed, just raw text
2. **Dense signal**: Learn from every token (vs one label per example)
3. **Compression = understanding**: To predict well, must understand syntax, semantics, facts, reasoning
4. **Scalable**: Can train on trillions of tokens (all of internet)

**Example:**
```
Input:  "The capital of France is"
Target: "capital of France is Paris"

Model must learn:
- Grammar (capital not capitals)
- World knowledge (France → Paris)
- Context (capital means city, not money)
```

**Why not other objectives?**

- **Masked LM (BERT)**: Predict random masked tokens - good for encoding, not generation
- **Denoising (T5)**: Predict corrupted spans - more complex, not clearly better
- **Contrastive (CLIP)**: Match text/images - different use case

**Gotcha**: Next-token prediction seems simple but enables emergent abilities like reasoning, few-shot learning at scale.

## Follow-up Questions
- How is this loss computed efficiently for long sequences?
- Why is cross-entropy the right loss function?
- What's the difference from masked language modeling?
