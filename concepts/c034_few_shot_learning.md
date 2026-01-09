# Few-Shot Learning & In-Context Learning

**Category:** evaluation
**Difficulty:** 3
**Tags:** prompting, evaluation, emergence

## Question
What is in-context learning and how does it differ from fine-tuning?

## What to Cover
- **Set context by**: Explaining that ICL requires no parameter updates—it's inference-time learning
- **Must mention**: The comparison with fine-tuning (speed, cost, persistence), zero/one/few-shot variants, emergence (only works at scale)
- **Show depth by**: Mentioning factors affecting performance (example selection, order, format) and gotchas (inconsistency, data leakage concerns)
- **Avoid**: Treating ICL as equivalent to fine-tuning or not mentioning the scale requirement

## Answer
**In-Context Learning (ICL)**: Model learns from examples in the prompt without parameter updates.

**Example:**
```
Translate English to French:
English: Hello → French: Bonjour
English: Goodbye → French: Au revoir
English: Thank you → French: [model completes: Merci]
```

**vs Fine-tuning:**

| Aspect | ICL | Fine-tuning |
|--------|-----|-------------|
| Parameters | Frozen | Updated |
| Examples needed | 0-100 | 100-100K+ |
| Speed | Instant | Hours/days |
| Persistence | Per-prompt | Permanent |
| Cost | Inference only | Training compute |

**Types:**

- **Zero-shot**: No examples, just instruction
  - "Translate to French: Hello"
- **One-shot**: One example
- **Few-shot**: 2-10 examples (standard)
- **Many-shot**: 100+ examples (limited by context window)

**Why it works (theories):**

1. **Bayesian inference**: Model infers task from examples
2. **Gradient descent in forward pass**: Attention implements implicit optimization
3. **Memorization**: Seen similar patterns during pretraining

**Emergence**: Only appears at scale (~10B+ parameters).

**Factors affecting ICL performance:**

1. **Model size**: Bigger = better ICL
2. **Example selection**: Relevant examples help
3. **Example order**: Different orders → different results
4. **Prompt format**: Phrasing matters a lot

**Gotchas:**

- **Inconsistent**: Small prompt changes → large performance swings
- **Data leakage**: Model might have seen test set during pretraining
- **Unfaithful**: Model might ignore examples (especially if they contradict pretraining)

**Modern use:** Most LLM applications use ICL (prompting), not fine-tuning.

## Follow-up Questions
- Why does ICL only work for large models?
- How do you select good few-shot examples?
- Is ICL really "learning" or just pattern matching?
