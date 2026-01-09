# LLM Safety & Alignment

**Category:** rlhf
**Difficulty:** 3
**Tags:** safety, alignment, evaluation

## Question
What are the main safety challenges for LLMs and how are they addressed?

## What to Cover
- **Set context by**: Categorizing safety challenges (harmful content, bias, privacy, misuse, overreliance)
- **Must mention**: Mitigation at training-time (data filtering, RLHF, Constitutional AI, red-teaming) and inference-time (guardrails, filtering)
- **Show depth by**: Discussing the capability-safety tension, jailbreaks, and open problems (steerable safety, robustness)
- **Avoid**: Only listing challenges without explaining mitigation strategies or discussing the tradeoffs

## Answer
**Safety challenges:**

**1. Harmful content generation**
- Violence, illegal activities, harassment
- Bioweapons, cyberattacks
- Misinformation

**2. Bias & fairness**
- Gender, race, religion stereotypes
- Learned from biased training data

**3. Privacy**
- Memorizing training data (personal info, copyrighted text)
- Regurgitating private data

**4. Misuse**
- Spam, phishing, social engineering
- Academic dishonesty
- Automated propaganda

**5. Overreliance & trust**
- Users trusting incorrect information
- Hallucinations presented confidently

**Mitigation strategies:**

**1. Training-time:**

**Data filtering:**
- Remove toxic content
- Remove PII (personally identifiable information)
- Balance demographic representation

**RLHF:**
- Train reward model on human preferences for safety
- Penalize harmful outputs

**Constitutional AI:**
- Self-critique using safety principles
- AI feedback for harmlessness

**Red-teaming:**
- Adversarial testing during training
- Find and patch failure modes

**2. Inference-time:**

**Input filtering:**
- Detect malicious prompts (jailbreaks)
- Block banned topics

**Output filtering:**
- Toxicity detection
- PII detection
- Fact-checking

**Guardrails:**
- Llama Guard, Nvidia NeMo Guardrails
- Rule-based + model-based filtering

**3. Evaluation:**

**Benchmarks:**
- TruthfulQA (truthfulness)
- BBQ (bias)
- RealToxicityPrompts (toxicity)

**Red-teaming:**
- Continuous adversarial testing
- Humans + automated attacks

**Adversarial robustness:**
- Jailbreaks (prompt injection)
- Many-shot jailbreaking
- Encoded attacks (Base64, leetspeak)

**Challenges:**

1. **Capability-safety tension**: More capable → harder to align
2. **Overfitting safety**: Refusing benign requests
3. **Cat-and-mouse**: New jailbreaks discovered constantly
4. **Cultural differences**: Safety norms vary by culture
5. **Measurement**: Hard to quantify "aligned"

**Refusal strategies:**
```
❌ "I can't help with that" (too sensitive)

✓ "I can't provide instructions for harmful activity X,
   but I can help with related safe topic Y"
```

**Modern approaches:**

- **Anthropic**: Constitutional AI + red-teaming
- **OpenAI**: RLHF + iterative deployment
- **Meta**: Llama Guard (separate safety model)
- **Industry**: Move toward "safety by design"

**Open problems:**
- Steerable safety (user control vs platform control)
- Robustness to adversarial prompts
- Truthfulness without over-refusing

## Follow-up Questions
- What is a jailbreak and how do you prevent it?
- How do you measure alignment?
- What's the capability-safety tradeoff?
