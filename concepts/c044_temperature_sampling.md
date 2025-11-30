# Temperature & Sampling Strategies

**Category:** modern_llm
**Difficulty:** 3
**Tags:** inference, generation, sampling

## Question
What is temperature in language model sampling and what are the main sampling strategies?

## Answer
**Temperature**: Scale logits before softmax to control randomness.

**Formula:**
```
p_i = exp(z_i / T) / Σ exp(z_j / T)

where:
  z_i = logit for token i
  T = temperature
```

**Effect:**
- **T = 0**: Greedy (always most likely token)
- **T < 1**: Less random, sharper distribution
- **T = 1**: Unmodified probabilities
- **T > 1**: More random, flatter distribution

**Example:**
```
Logits: [4.0, 3.0, 2.0]

T=0.5 (focused):
  Probs: [0.70, 0.24, 0.06]

T=1.0 (neutral):
  Probs: [0.59, 0.24, 0.10]

T=2.0 (random):
  Probs: [0.42, 0.31, 0.23]
```

**Sampling strategies:**

**1. Greedy decoding** (T=0)
- Always pick most likely token
- Deterministic, repetitive
- Good for: Factual QA, extraction

**2. Pure sampling** (T=1)
- Sample from full distribution
- Too random for most tasks

**3. Top-k sampling**
```
Keep top k tokens, renormalize, sample
k=50 is typical
```
- Prevents sampling very unlikely tokens
- Still can sample bad tokens if top-k all bad

**4. Top-p (nucleus) sampling**
```
Keep smallest set where cumulative prob > p
p=0.9 is typical
```
- Adaptive: k varies by uncertainty
- High confidence → small set, low confidence → larger set
- **Most common in modern LLMs**

**5. Min-p sampling**
```
Keep tokens with prob > p * max_prob
p=0.1 typical
```
- Alternative to top-p
- More intuitive threshold

**Hybrid: Top-p + Temperature**
```
1. Apply temperature to logits
2. Apply top-p filtering
3. Sample from filtered distribution

Example: T=0.7, p=0.9 (ChatGPT-like)
```

**Use cases:**
- **Creative writing**: T=0.8-1.0, top-p=0.95
- **Code generation**: T=0.2-0.5, top-p=0.9
- **Factual QA**: T=0.0 (greedy)
- **Chat**: T=0.7, top-p=0.9

## Follow-up Questions
- Why use top-p instead of top-k?
- What happens with very high temperature?
- How do you choose temperature for a task?
