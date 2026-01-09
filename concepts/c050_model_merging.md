# Model Merging & Weight Averaging

**Category:** modern_llm
**Difficulty:** 3
**Tags:** training, optimization, ensembling

## Question
What is model merging and when is it useful?

## What to Cover
- **Set context by**: Explaining model merging as combining weights from multiple models without retraining
- **Must mention**: Methods (linear interpolation, task arithmetic, TIES), use cases (multi-task, model soups), when it works well (same architecture, same initialization)
- **Show depth by**: Mentioning real examples (community models on HuggingFace) and gotchas (layer norm, embeddings)
- **Avoid**: Only describing linear averaging without mentioning more sophisticated methods or limitations

## Answer
**Model Merging**: Combine weights from multiple models to create a single model.

**Basic approach (Linear interpolation):**
```
θ_merged = α * θ_1 + (1-α) * θ_2

Example:
θ_merged = 0.5 * θ_base + 0.5 * θ_finetuned
```

**Methods:**

**1. Model Soups (Uniform averaging)**
```
Train N models with different hyperparameters/seeds
θ_soup = (1/N) * Σ θ_i

Result: Often better than any individual model!
```

**2. SLERP (Spherical interpolation)**
```
Instead of linear interpolation, interpolate on hypersphere
Better preserves model geometry
Used for merging LoRA adapters
```

**3. Task Arithmetic**
```
θ_task = θ_base + λ * (θ_finetuned - θ_base)

Can combine multiple tasks:
θ_multi = θ_base + λ_1*(θ_task1 - θ_base) + λ_2*(θ_task2 - θ_base)
```

**4. TIES (Trim, Elect, Merge)**
```
1. Trim: Remove small weight changes
2. Elect: Resolve sign conflicts
3. Merge: Combine remaining weights

Better than naive averaging
```

**Use cases:**

**Stochastic Weight Averaging (SWA):**
- Average checkpoints from end of training
- Improves generalization
- Common in training pipelines

**Multi-task merging:**
```
Base model: Llama-7B
Task 1: Math fine-tune
Task 2: Code fine-tune

Merged: Gets both math + code abilities
```

**Community models (HuggingFace):**
- Merge open-source models with different strengths
- "Frankenstein" models (e.g., Goliath = Llama + Falcon merge)

**When it works well:**

✓ Models trained from same initialization
✓ Similar architectures
✓ Complementary capabilities
✓ Same tokenizer/vocabulary

**When it fails:**

✗ Very different architectures
✗ Different training objectives
✗ Conflicting capabilities

**Gotchas:**

1. **Compatibility**: Models must have same architecture
2. **Layer norm**: Sometimes exclude from merging
3. **Embedding layers**: Be careful with vocabulary differences
4. **Quality**: Simple averaging often works surprisingly well

**Example (real):**
```
Nous-Hermes-13B = Merge of:
  - Llama-2-13B (base)
  - Llama-2-13B-chat (instruction following)
  - CodeLlama-13B (code)

Result: Good at general chat + code
```

## Follow-up Questions
- Why does averaging models improve generalization?
- How do you choose mixing weights?
- Can you merge models with different architectures?
