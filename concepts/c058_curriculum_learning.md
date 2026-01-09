# Curriculum Learning & Data Ordering

**Category:** training
**Difficulty:** 3
**Tags:** training, data, optimization

## Question
What is curriculum learning and does data ordering matter for training LLMs?

## What to Cover
- **Set context by**: Explaining curriculum learning as training on easier examples first
- **Must mention**: How to define difficulty (length, perplexity), when it helps (small data, domain-specific), modern finding (data quality > ordering for large-scale training)
- **Show depth by**: Discussing practical guidance (matters less for trillion-token training, more for fine-tuning) and anti-curriculum as alternative
- **Avoid**: Overselling curriculum learning—modern research shows diminishing returns at scale

## Answer
**Curriculum Learning**: Train on easier examples first, gradually increase difficulty.

**Analogy**: Like human education - learn addition before calculus.

**Standard approach:**
```
Shuffle data randomly each epoch
All examples treated equally
```

**Curriculum learning:**
```
Epoch 1: Easy examples (short, simple)
Epoch 2: Medium examples
Epoch 3: Hard examples (long, complex)
```

**How to define "difficulty":**

1. **Length**: Shorter sequences → easier
2. **Perplexity**: Lower perplexity (under pretrained model) → easier
3. **Task-specific**: For math, basic arithmetic → calculus
4. **Domain**: Common topics → rare topics

**Benefits:**

✓ **Faster convergence**: Model learns basic patterns first
✓ **Better final performance**: In some domains (especially small data)
✓ **Stability**: Less likely to diverge early

**Challenges:**

✗ **Defining difficulty**: Not always clear what's "easy"
✗ **Extra complexity**: Need to score/sort data
✗ **Diminishing returns**: Benefit smaller with large datasets

**For LLM pretraining:**

**Data ordering matters less than expected:**
- Random shuffling works well
- Huge datasets → see all difficulties naturally
- Some evidence that quality filtering > ordering

**Exception - Domain/task-specific:**
```
Code LLMs: Python → C++ → Assembly
Math LLMs: Arithmetic → Algebra → Calculus
Multilingual: High-resource → Low-resource languages
```

**Token-level vs example-level:**

**Example-level**: Sort entire documents
**Token-level**: Within document, predict easy tokens first
- Less explored, promising

**Deduplication as implicit curriculum:**
- Remove duplicates → model sees rare examples more
- Similar effect to curriculum (focus on harder, rarer examples)

**Anti-curriculum (sometimes better!):**
```
Start with hard examples
Rationale: Push model to learn complex patterns early
Mixed evidence
```

**Modern findings (Scaling Laws papers):**

- Data quality > data ordering
- For trillion-token training, ordering negligible
- Better to focus on filtering bad data

**Practical advice:**

1. **Small datasets** (<10M examples): Try curriculum
2. **Large datasets** (>1B tokens): Random shuffle fine
3. **Domain-specific**: Curriculum can help
4. **Finetuning**: Curriculum more helpful (smaller scale)

**Example (Codex):**
```
Phase 1: Clean GitHub repos (high quality)
Phase 2: All GitHub (medium quality)
Phase 3: StackOverflow (Q&A format)

Progressive difficulty in code complexity
```

## Follow-up Questions
- How do you measure example difficulty?
- Does curriculum learning help for large-scale pretraining?
- What's the difference between curriculum and data filtering?
