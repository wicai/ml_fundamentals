# Fine-Tuning Strategies & Best Practices

**Category:** training
**Difficulty:** 3
**Tags:** finetuning, training, optimization

## Question
What are the different strategies for fine-tuning LLMs and when should you use each?

## Answer
**Fine-tuning**: Adapt pretrained model to specific task/domain.

**Main strategies:**

**1. Full fine-tuning**
```
Update all parameters
Highest quality, most expensive

When to use:
  ✓ Large dataset (10K+ examples)
  ✓ Very different domain (medical, legal)
  ✓ Have compute budget
```

**Hyperparameters:**
```
Learning rate: 1e-5 to 5e-5 (lower than pretraining)
Epochs: 2-5 (avoid overfitting)
Batch size: 8-32
Warmup: 5-10% of steps
Weight decay: 0.01
```

**2. LoRA (Low-Rank Adaptation)**
```
Freeze base model, add trainable adapters
1% of parameters, 90-95% of quality

When to use:
  ✓ Limited compute
  ✓ Multiple tasks (swap adapters)
  ✓ Medium dataset (1K-10K examples)
```

**Hyperparameters:**
```
Rank (r): 8-64 (16 is sweet spot)
Alpha: 16-32 (often 2*r)
Target modules: Q,V or Q,K,V,O
Learning rate: 1e-4 to 3e-4
```

**3. QLoRA (Quantized LoRA)**
```
LoRA + 4-bit quantization of base model
Fine-tune 70B on single GPU!

When to use:
  ✓ Very limited GPU memory
  ✓ Large models (70B+)
```

**4. Prefix tuning / P-tuning**
```
Freeze model, train soft prompt embeddings

Less effective than LoRA
Rarely used now
```

**5. Adapter layers**
```
Insert small trainable layers between frozen layers

Similar to LoRA but older approach
LoRA generally preferred
```

**By task type:**

**Instruction following:**
```
Data format: (instruction, output) pairs
Objective: Cross-entropy on output tokens
Dataset size: 10K-100K examples

Example:
  Instruction: "Summarize this article..."
  Output: "The article discusses..."
```

**Classification:**
```
Option 1: Generate class name
  "Sentiment: positive/negative/neutral"

Option 2: Add classification head
  Freeze LLM, train linear layer on [CLS]
```

**Code generation:**
```
Data: (docstring, code) or (test, code)
Often benefits from larger rank LoRA (32-64)
```

**Domain adaptation:**
```
Continue pretraining on domain corpus
Then fine-tune on task

Example: Medical LLM
  1. Pretrain on PubMed (100B tokens)
  2. Fine-tune on clinical notes (10K examples)
```

**Data considerations:**

**Quality > quantity:**
```
1K high-quality examples > 10K noisy examples
Invest in data curation
```

**Data format:**
```
Instruction format (common):
  <s>[INST] {instruction} [/INST] {response}</s>

Or chat format:
  <|user|>{message}<|assistant|>{response}
```

**Data augmentation:**
```
- Paraphrase instructions (GPT-4)
- Back-translation
- Synthetic data generation
```

**Preventing catastrophic forgetting:**

**Replay:**
```
Mix task data with general data
80% task-specific, 20% general
```

**Regularization:**
```
Higher weight decay
KL divergence penalty from base model
```

**Early stopping:**
```
Monitor validation loss on general tasks
Stop if general performance degrades
```

**Learning rate strategies:**

**Constant with warmup:**
```
Most common for fine-tuning
LR warmup (5%), then constant
```

**Cosine decay:**
```
For longer fine-tuning (many epochs)
Smooth convergence
```

**Layer-wise LR decay:**
```
Lower LR for early layers, higher for late layers
Early layers = general features
Late layers = task-specific
```

**Evaluation:**

**Task metrics:**
- Accuracy, F1, BLEU, ROUGE (task-dependent)

**General capability:**
- Test on MMLU, HumanEval (ensure no regression)

**Human eval:**
- For generation tasks, automated metrics insufficient

**Best practices:**

1. **Start with LoRA**: Cheaper, iterate faster
2. **Small LR**: 10× smaller than pretraining
3. **Few epochs**: 2-3 usually enough
4. **Monitor overfitting**: Validation loss
5. **Mix data**: Prevent catastrophic forgetting
6. **Use chat template**: Consistent formatting
7. **Evaluate broadly**: Task + general performance

## Follow-up Questions
- How do you choose between full fine-tuning and LoRA?
- What learning rate should you use for fine-tuning?
- How much data do you need for fine-tuning?
