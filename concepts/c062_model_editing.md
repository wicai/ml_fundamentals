# Model Editing & Knowledge Updating

**Category:** modern_llm
**Difficulty:** 4
**Tags:** knowledge, editing, inference

## Question
How can you update specific facts in an LLM without full retraining?

## Answer
**Problem**: Model has outdated or wrong fact.
```
"The president of the US is Donald Trump" (outdated)
"Paris is the capital of Germany" (wrong)
```

**Goal**: Fix specific fact without retraining or degrading other knowledge.

**Approaches:**

**1. Fine-tuning (naive)**
```
Create dataset: "The president is Joe Biden"
Fine-tune model

Problems:
  - Catastrophic forgetting (hurts other knowledge)
  - Doesn't guarantee fact learned
  - Expensive (full training run)
```

**2. Constrained fine-tuning**
```
Fine-tune with regularization to preserve other knowledge

Loss = BCE(new_fact) + λ * KL(new_model || old_model)

Better but still expensive
```

**3. Locate-and-edit (MEMIT, ROME)**

**Key insight**: Facts stored in specific MLP layers (feed-forward)

**Algorithm:**
```
1. Locate where fact is stored:
   - Run model on "The president of the US is"
   - Find which layers activate for "president of US"
   - Typically mid-layer MLPs

2. Compute what change needed:
   - Want output to change from "Trump" to "Biden"
   - Compute gradient ∂output/∂weights

3. Directly edit weights:
   - Update W → W + ΔW
   - ΔW computed to change this fact minimally
```

**ROME (Rank-One Model Editing):**
```
Find key-value pair in MLP layer
  Key: "president of the US"
  Value: "Joe Biden"

Update single layer's weights to store new association
```

**MEMIT (Mass Editing Memory In Transformer):**
- Extension of ROME for multiple facts
- Edit thousands of facts simultaneously

**4. External memory / RAG**
```
Don't edit model, use retrieval:

Query: "Who is the president?"
Retrieve: Latest fact from database
Generate: Use retrieved fact in context

Pros: Easy to update, no model changes
Cons: Latency, need retrieval system
```

**5. Meta-learning approaches**
```
Train model to be "editable"
Special fine-tuning that makes future edits easier

Hypernetworks, meta-gradients, etc.
Research area
```

**Evaluation criteria:**

1. **Efficacy**: Does edit work?
   - "Who is president?" → "Joe Biden" ✓

2. **Specificity**: Doesn't change unrelated facts
   - "What is the capital of France?" → Still "Paris" ✓

3. **Generalization**: Related queries work
   - "The US president is..." → "Joe Biden" ✓
   - "Biden is the president of..." → "United States" ✓

**Challenges:**

1. **Ripple effects**: Changing one fact affects related facts
   - Update: "Biden is president"
   - Should also update: "Who lives in White House?"

2. **Multiple locations**: Facts stored in multiple places
   - Attention layers, MLPs, embeddings

3. **Verification**: Hard to know if edit succeeded comprehensively

4. **Scale**: Editing thousands of facts is hard

**When to use what:**

| Method | Few facts | Many facts | Real-time | Guarantees |
|--------|-----------|------------|-----------|------------|
| Fine-tune | ✗ | ✓ | ✗ | Weak |
| ROME/MEMIT | ✓ | ✓ | ✓ | Medium |
| RAG | ✓ | ✓ | ✓ | Strong |

**Modern practice:**
- **RAG** for dynamic/frequently changing facts
- **MEMIT** for permanent model updates
- **Fine-tuning** for major knowledge updates

**Open problems:**
- Editing reasoning abilities (not just facts)
- Verifying edits don't break anything
- Scaling to millions of edits

## Follow-up Questions
- Where are facts stored in transformer models?
- How does ROME work at a high level?
- What's better: model editing or RAG?
