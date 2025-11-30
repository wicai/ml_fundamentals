# Hallucination in LLMs

**Category:** evaluation
**Difficulty:** 3
**Tags:** hallucination, safety, quality

## Question
What causes LLMs to hallucinate and how can you mitigate it?

## Answer
**Hallucination**: Model generates plausible-sounding but factually incorrect or nonsensical output.

**Types:**

**1. Factual hallucination:**
```
Q: "Who won the 2024 World Cup?"
A: "Brazil won the 2024 World Cup" (made up)

Model generates confident but wrong fact
```

**2. Intrinsic hallucination:**
```
Contradicts source material

Source: "The company has 100 employees"
Summary: "The company has 500 employees"
```

**3. Extrinsic hallucination:**
```
Adds information not in source

Source: "The company was founded in 2020"
Summary: "The company was founded in 2020 in San Francisco by John Smith"

(Location and founder not mentioned)
```

**Why hallucination happens:**

**1. Training objective:**
```
Trained to predict plausible next token
NOT trained to be factual

"The capital of France is" → "Paris" (memorized)
"The capital of Atlantis is" → Makes up plausible answer
```

**2. Parametric knowledge limits:**
```
Knowledge cutoff (e.g., 2023)
Rare facts not in training data
Model doesn't know, but still generates

Better to say "I don't know" but model trained to complete
```

**3. Overgeneralization:**
```
Learns patterns, applies incorrectly

Pattern: "X won Y award in 2023"
Generates: "Alice Smith won Nobel Prize in 2023"
(Even if false)
```

**4. Prompt ambiguity:**
```
Unclear prompt → model fills in details

"Tell me about the meeting" (which meeting?)
Model invents plausible meeting details
```

**5. Optimization for plausibility:**
```
RLHF optimizes for "sounding good"
Not for factual accuracy

Confident wrong answer > uncertain right answer
(From model's perspective)
```

**Detection methods:**

**1. Consistency checks:**
```
Generate multiple answers
If answers disagree → likely hallucinating

Self-consistency: Majority vote
```

**2. Confidence estimation:**
```
Softmax probability of generated tokens
Low probability → less confident

Challenge: Model can be confidently wrong
```

**3. External verification:**
```
Check facts against knowledge base
Retrieve supporting evidence
Flag if no support found
```

**4. Contrastive decoding:**
```
Compare against weaker model
Unique to strong model but not weak → likely hallucination
```

**5. Trained verifiers:**
```
Separate model to verify factuality
Takes claim + evidence → true/false
```

**Mitigation strategies:**

**1. Retrieval-Augmented Generation (RAG):**
```
Retrieve facts before generating
Ground generation in retrieved docs

Significantly reduces hallucination
```

**2. Instruction tuning:**
```
Fine-tune to say "I don't know"
Include examples of refusing to answer

"I'm not sure about that" vs making up facts
```

**3. Chain-of-thought verification:**
```
Generate reasoning before answer
Self-critique and revise

Prompt: "Think step by step and verify your answer"
```

**4. Citation forcing:**
```
Require model to cite sources

"According to [source], ..."
Can't cite → don't generate
```

**5. Temperature reduction:**
```
Lower temperature → less creative, more conservative
Reduces hallucination but also creativity
```

**6. Constrained decoding:**
```
Limit outputs to known entities
Use knowledge graph to constrain

Can't generate entity not in KB
```

**7. Human-in-the-loop:**
```
Human verification before deployment
Fact-checking critical outputs
```

**8. Uncertainty communication:**
```
Express confidence levels

"I'm confident that..."
"I'm uncertain, but..."
"I don't have information about..."
```

**Evaluation benchmarks:**

**TruthfulQA:**
- Questions designed to elicit false answers
- Measures truthfulness
- GPT-4: ~60% (still room for improvement)

**HaluEval:**
- Detect hallucinations in QA, summarization
- Binary classification: hallucinated or not

**FActScore:**
- Atomic fact verification
- Break response into facts, verify each

**Best practices:**

1. **Use RAG** for factual queries
2. **Lower temperature** for factual tasks
3. **Request citations** in prompts
4. **Verify critical outputs** with external sources
5. **Communicate uncertainty** to users
6. **Fine-tune** with "I don't know" examples

**Open problems:**

- Models don't know what they don't know
- Confident hallucinations hard to detect
- Trade-off: accuracy vs helpfulness

## Follow-up Questions
- Why are LLMs confidently wrong?
- How does RAG reduce hallucination?
- Can you eliminate hallucination entirely?
