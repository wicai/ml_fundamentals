# Data Filtering & Quality

**Category:** training
**Difficulty:** 3
**Tags:** data, training, pretraining

## Question
How do you filter and curate data for pretraining LLMs? What makes "high-quality" data?

## Answer
**Problem**: Internet data is noisy, toxic, copyrighted, low-quality.

**Filtering pipeline:**

**1. Deduplication**
- Remove exact duplicates (common in web scrapes)
- MinHash LSH for near-duplicates
- Reduces dataset by 20-50%
- Why: Prevents memorization, improves generalization

**2. Quality filtering**

**Heuristics:**
- Length: Too short (<50 chars) or too long (>100k chars)
- Language: Filter to desired languages
- Repetition: High n-gram repetition → spam/low quality
- Special characters: Too many → not natural language
- Perplexity: Use smaller model to score, filter high-perplexity (gibberish)

**Classifier-based:**
- Train binary classifier (quality vs not) on labeled subset
- Score all documents, keep top X%
- Example: FastText on web pages

**3. Toxicity & Safety**
- Toxic language detection (Perspective API, custom models)
- PII detection (emails, phone numbers, SSNs)
- Trade-off: Aggressive filtering might reduce diversity

**4. Copyright & Legal**
- Books3 controversy (copyrighted books)
- GitHub Copilot (code licensing)
- Many models now exclude copyrighted content

**5. Data mixing (multi-source)**
```
CommonCrawl (web): 60%
Books: 20%
Wikipedia: 10%
GitHub (code): 5%
ArXiv (papers): 5%
```

Ratios are tuned hyperparameters!

**Quality indicators:**

- Human-written (not machine-generated)
- Informative (not SEO spam)
- Coherent (grammar, structure)
- Diverse (topics, styles)
- Factual (not misinformation)

**Modern approaches:**

**DataComp / LAION**
- Systematic data filtering experiments
- Finding: Aggressive quality filtering helps even if dataset shrinks

**Phi models (Microsoft)**
- Extreme quality filtering ("textbook-quality" data)
- Smaller dataset, better results

**Gotcha**: Data quality matters more than quantity. 100B high-quality tokens > 1T noisy tokens.

## Follow-up Questions
- How do you detect near-duplicates efficiently?
- What's the toxicity vs diversity trade-off?
- Should you include machine-generated text in pretraining data?
