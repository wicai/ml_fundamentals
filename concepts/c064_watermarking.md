# LLM Output Watermarking

**Category:** modern_llm
**Difficulty:** 3
**Tags:** safety, detection, generation

## Question
How can you watermark LLM-generated text to detect it later?

## What to Cover
- **Set context by**: Explaining use cases (detecting AI content, attribution, academic integrity)
- **Must mention**: The main algorithm (green/red list based on previous token), detection via statistical test, properties (imperceptible, robust, removable)
- **Show depth by**: Discussing evasion attacks (paraphrasing, encoding) and limitations (open models, short text)
- **Avoid**: Only describing watermarking without discussing its limitations and the arms race

## Answer
**Goal**: Embed imperceptible signal in generated text that proves it came from your model.

**Use cases:**
- Detect AI-generated content (misinformation, spam)
- Attribution (which model generated this?)
- Prevent academic dishonesty
- Content moderation

**Main approach (Kirchenbauer et al.):**

**Watermarking algorithm:**
```
For each token generation:

1. Use previous token to seed RNG:
   seed = hash(previous_token)

2. Partition vocabulary into green list (favored) and red list:
   green_list = random_subset(vocabulary, seed)
   red_list = vocabulary \ green_list

3. Boost green list tokens before sampling:
   logits[green_tokens] += δ  # δ = 2-5 typical

4. Sample from modified distribution

Result: Text has statistically more green tokens than random text
```

**Detection:**
```
For each token in text:
  seed = hash(previous_token)
  green_list = random_subset(vocabulary, seed)
  count += 1 if current_token in green_list else 0

z-score = (count - expected) / std
if z-score > threshold:
  return "WATERMARKED"
```

**Statistical test:**
```
Random text: ~50% green tokens
Watermarked text: ~60-70% green tokens

With 100 tokens, can detect with high confidence
```

**Properties:**

✓ **Imperceptible**: Doesn't hurt text quality (small δ)
✓ **Robust**: Survives paraphrasing, minor edits
✓ **Zero-knowledge**: Don't need model access to detect
✗ **Removable**: Adversary can remove with effort (paraphrase heavily)
✗ **Short text**: Needs ~100 tokens for reliable detection

**Advanced techniques:**

**1. Multi-bit watermarking:**
- Encode message in watermark (e.g., model ID, timestamp)
- Multiple color lists per bit

**2. Semantic watermarking:**
- Watermark meaning, not just tokens
- More robust to paraphrasing

**3. Undetectable watermarking:**
- Cryptographic guarantees of undetectability
- Trade-off: weaker signal, needs more tokens

**Evasion attacks:**

1. **Paraphrasing**: Rewrite text
   - Defense: Semantic watermarking

2. **Translation**: Translate to another language and back
   - Defense: Cross-lingual watermarking

3. **Copying**: Mix AI text with human text
   - Defense: Detect segments

4. **Substitution**: Replace random tokens
   - Defense: Robust to X% substitution

**Limitations:**

1. **False positives**: Human text can randomly match
   - Mitigated by threshold tuning

2. **Open models**: Can't watermark if anyone can run model
   - Only works for API-based models

3. **Incentive**: Users want to remove watermarks
   - Arms race

4. **Quality degradation**: Watermark strength vs quality trade-off

**Alternatives to watermarking:**

**1. Retrieval-based detection:**
- Store all generated texts
- Check if new text matches
- Privacy issues

**2. Statistical detection (GPTZero, DetectGPT):**
- Train classifier to detect AI text
- Doesn't require watermark
- Arms race (bypass with paraphrasing)

**3. Model fingerprinting:**
- Each model has unique statistical signature
- Detect which model generated text

**Deployment challenges:**

- **Adoption**: Need all LLM providers to watermark
- **Standardization**: Common watermarking scheme
- **Education**: Users need to understand detection isn't perfect

## Follow-up Questions
- How robust is watermarking to paraphrasing?
- Can you watermark without degrading quality?
- What if adversary knows the watermarking scheme?
