# Tokenization for LLMs

**Category:** modern_llm
**Difficulty:** 3
**Tags:** tokenization, preprocessing, architecture

## Question
What is BPE/SentencePiece tokenization and why is it used instead of word-level tokens?

## What to Cover
- **Set context by**: Explaining the tradeoffs between word-level (OOV problem), character-level (too long), and subword
- **Must mention**: The BPE algorithm (iterative merging), why it handles OOV, morphological benefits
- **Show depth by**: Naming variants (SentencePiece, WordPiece, Unigram) and noting tokenization differs across models
- **Avoid**: Describing only BPE without mentioning the problems it solves or modern variants

## Answer
**Byte-Pair Encoding (BPE):**
Iteratively merge the most frequent pair of tokens to build a vocabulary.

**Algorithm:**
1. Start with character-level tokens
2. Find most frequent adjacent pair (e.g., "t" + "h" → "th")
3. Merge into new token
4. Repeat until vocabulary size reached (typically 32K-100K tokens)

**Why not word-level?**

1. **OOV problem**: Unknown words require special <UNK> token
2. **Vocabulary explosion**: English has >170K words, multiply by languages
3. **Morphology**: "play", "playing", "played" are separate words but related

**Why not character-level?**

1. **Long sequences**: "Transformer" = 11 tokens vs 1 subword token
2. **Computational cost**: Attention is O(n²), longer sequences are expensive
3. **Weaker semantics**: Harder to learn meaning from individual characters

**BPE advantages:**

- **No OOV**: Can encode any text with byte-level fallback
- **Compression**: Common words = 1 token, rare words = few tokens
- **Morphology**: "play" + "ing" shares "play" subword across forms

**Modern variants:**
- **SentencePiece**: Language-agnostic (no pre-tokenization)
- **WordPiece**: Used by BERT (slightly different merging criterion)
- **Unigram**: Used by T5 (probabilistic approach)

**Gotcha**: Tokenization is **not** deterministic across models! GPT-3 and LLaMA have different vocabularies.

## Follow-up Questions
- How does tokenization affect non-English languages?
- What's the impact of vocabulary size on model performance?
- Why do some characters get split into multiple tokens?
