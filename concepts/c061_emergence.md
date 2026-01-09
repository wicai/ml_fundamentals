# Emergent Abilities in LLMs

**Category:** evaluation
**Difficulty:** 3
**Tags:** scaling, emergence, capabilities

## Question
What are emergent abilities in LLMs and what causes them?

## What to Cover
- **Set context by**: Defining emergence (abilities appearing suddenly at scale, not present in smaller models)
- **Must mention**: Examples (arithmetic, few-shot learning, CoT), hypotheses for why emergence happens (thresholds, measurement artifacts)
- **Show depth by**: Discussing the debate (Schaeffer et al. arguing it's a measurement artifact) and practical implications (unpredictability, safety concerns)
- **Avoid**: Treating emergence as settled science—it's actively debated

## Answer
**Emergent Abilities**: Capabilities that appear suddenly at scale, not present in smaller models.

**Definition (Wei et al.):**
An ability is emergent if:
- Not present in small models
- Appears unpredictably at a certain scale
- Performance jumps sharply (not smooth)

**Examples:**

**Arithmetic:**
- <1B params: Can't add 3-digit numbers
- 10B+ params: Can add multi-digit numbers
- Appears suddenly around 10B

**Few-shot learning:**
- <10B: Minimal benefit from examples
- 100B+: Strong few-shot learning
- Emerges around 10-100B scale

**Chain-of-thought reasoning:**
- <10B: CoT hurts performance (confuses model)
- 100B+: CoT helps significantly
- Threshold around 10B params

**Other emergent abilities:**
- Multi-step reasoning
- Following complex instructions
- Code generation
- Translation for low-resource languages

**Hypotheses for why emergence:**

**1. Scaling laws with thresholds**
```
Accuracy improves smoothly with scale
But task has accuracy threshold (e.g., 50% to be useful)
Looks like sudden "emergence" but is smooth crossing of threshold
```

**2. Benchmark artifacts**
- Metrics like accuracy are discrete
- Continuous improvement in logits/probabilities
- Appears sudden due to metric choice

**3. Compression & memorization**
```
Small models: Memorize specific patterns
Large models: Compress knowledge into general rules
Threshold where compression becomes possible
```

**4. Representation learning**
```
Certain abstractions require minimum capacity
Once capacity reached, abstractions form
Enables new capabilities
```

**Debate: Are abilities truly emergent?**

**Schaeffer et al. (2023) argue:**
- "Emergence" is measurement artifact
- Smooth metrics (like log-probability) show continuous improvement
- Discrete metrics (like accuracy) show apparent jumps

**Example:**
```
Multiple choice (4 options):
  Random guessing: 25%
  Model slightly better than random: 26-30% (looks bad)
  Model good: 75%+ (suddenly "works")

Perplexity improves smoothly: 100 → 50 → 20 → 10
Accuracy jumps: 25% → 25% → 80% (emergence!)
```

**Counter-argument:**
- Some abilities genuinely sudden (e.g., instruction following)
- Not explained by metrics alone

**Practical implications:**

1. **Hard to predict**: Can't know what abilities will emerge
2. **Scaling uncertainty**: Don't know what threshold needed
3. **Safety concern**: New capabilities might include harmful ones
4. **Evaluation**: Need continuous metrics to see smooth progress

**Examples in practice:**

**GPT-3** (175B):
- Emergent few-shot learning
- In-context learning without fine-tuning

**PaLM** (540B):
- Emergent chain-of-thought
- Multi-step reasoning

**GPT-4**:
- More reliable instruction following
- Better coding
- Multi-modal understanding

**Open question**: Will emergence continue at larger scales or plateau?

## Follow-up Questions
- Is emergence real or a measurement artifact?
- What abilities might emerge at 1T+ parameters?
- How do you predict emergent abilities?
