# Implement BPE Tokenizer

**Category:** coding
**Difficulty:** 3
**Tags:** coding, tokenization, bpe, language model

## Question
Implement Byte-Pair Encoding (BPE) tokenization — the algorithm used by GPT, LLaMA, and most modern LLMs.

**Important: This is word-level BPE (original paper style), not flat character-level BPE.**

### How word-level BPE works:
1. **Split the corpus into words** (on whitespace). Track word frequencies using a Counter.
2. **Represent each word as a tuple of characters + an end-of-word marker `</w>`**. For example, `"low"` becomes `("l", "o", "w", "</w>")`. The `</w>` marker preserves word boundaries so the decoder knows where to insert spaces.
3. **Iterate `num_merges` times:**
   - Count all adjacent pairs across all words, weighted by word frequency
   - Find the most frequent pair
   - Merge that pair in every word (replace consecutive tokens with their concatenation)
   - Record the pair as a merge rule
4. **Return the ordered list of merge rules**

### Encoding (applying merges to new text):
- Split text into words, represent each as characters + `</w>`
- Apply each merge rule in learned order (left to right within each word)

### Decoding:
- Join all tokens, replace `</w>` with spaces, strip trailing space

### Vocabulary:
- Base characters: all unique characters found by decomposing each side of every merge rule into individual chars
- Plus each merged token (`a + b` for each merge pair)
- Return `sorted(base_chars) + [a+b for a,b in merges]`

Your implementation should include:
1. **`learn_bpe`**: Learn merge rules from a training corpus
2. **`BPETokenizer`**: Tokenizer class that encodes and decodes text

**Function signature:**
```python
def learn_bpe(corpus: str, num_merges: int) -> list[tuple[str, str]]:
    """
    Learn BPE merge rules from a training corpus (word-level with </w> markers).

    Split corpus into words, represent each as characters + '</w>' marker,
    then repeatedly find the most frequent adjacent pair (weighted by word
    frequency) and merge them.

    Args:
        corpus: training text
        num_merges: number of merge operations to learn
    Returns:
        merges: ordered list of (token_a, token_b) pairs to merge,
                in the order they were learned
    """
    pass

class BPETokenizer:
    """
    Tokenizer using Byte-Pair Encoding.
    """
    def __init__(self, merges: list[tuple[str, str]]) -> None:
        """
        Args:
            merges: ordered list of merge rules from learn_bpe()
        """
        pass

    def encode(self, text: str) -> list[str]:
        """
        Encode text into BPE tokens.

        Split text into words, represent each as characters + '</w>',
        then apply merge rules in order.

        Args:
            text: input string
        Returns:
            tokens: list of token strings (including '</w>' markers)
        """
        pass

    def decode(self, tokens: list[str]) -> str:
        """
        Decode BPE tokens back to text.

        Join tokens, replace '</w>' with spaces, strip trailing space.

        Args:
            tokens: list of token strings
        Returns:
            text: decoded string
        """
        pass

    @property
    def vocab(self) -> list[str]:
        """Return the full vocabulary: sorted base chars (from merge rules) + merged tokens."""
        pass
```

## Answer

**Key concepts:**
1. Start with character-level tokens (each character is its own token)
2. Count all adjacent token pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat for num_merges iterations
5. Encoding: split text into characters, then apply merges in learned order
6. Common words get merged into single tokens; rare words stay as characters

**Reference implementation:**
```python
from collections import Counter

def learn_bpe(corpus: str, num_merges: int) -> list[tuple[str, str]]:
    """Learn BPE merge rules from corpus."""
    # Split corpus into words, represent each as a tuple of characters
    # Add end-of-word marker to distinguish word boundaries
    words = corpus.split()
    word_freqs: dict[tuple[str, ...], int] = Counter()
    for word in words:
        # Each word is a tuple of characters + end-of-word marker
        word_freqs[tuple(word) + ('</w>',)] += 1

    merges: list[tuple[str, str]] = []

    for _ in range(num_merges):
        # Count all adjacent pairs
        pair_counts: dict[tuple[str, str], int] = Counter()
        for word_tokens, freq in word_freqs.items():
            for i in range(len(word_tokens) - 1):
                pair = (word_tokens[i], word_tokens[i + 1])
                pair_counts[pair] += freq

        if not pair_counts:
            break

        # Find the most frequent pair
        best_pair = pair_counts.most_common(1)[0][0]
        merges.append(best_pair)

        # Merge that pair in all words
        new_word_freqs: dict[tuple[str, ...], int] = {}
        merged_token = best_pair[0] + best_pair[1]

        for word_tokens, freq in word_freqs.items():
            new_tokens: list[str] = []
            i = 0
            while i < len(word_tokens):
                if (i < len(word_tokens) - 1 and
                    word_tokens[i] == best_pair[0] and
                    word_tokens[i + 1] == best_pair[1]):
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1
            new_word_freqs[tuple(new_tokens)] = freq

        word_freqs = new_word_freqs

    return merges

class BPETokenizer:
    def __init__(self, merges: list[tuple[str, str]]) -> None:
        self.merges = merges
        # Build vocab: start with unique characters from merges, add merged tokens
        chars: set[str] = set()
        for a, b in merges:
            for c in a:
                chars.add(c)
            for c in b:
                chars.add(c)
        self._vocab = sorted(chars) + [a + b for a, b in merges]

    def encode(self, text: str) -> list[str]:
        """Apply learned merges to tokenize text."""
        # Split into words and process each
        words = text.split()
        all_tokens: list[str] = []

        for word_idx, word in enumerate(words):
            # Start with characters + end-of-word marker
            tokens = list(word) + ['</w>']

            # Apply each merge rule in order
            for pair_a, pair_b in self.merges:
                merged = pair_a + pair_b
                i = 0
                new_tokens: list[str] = []
                while i < len(tokens):
                    if (i < len(tokens) - 1 and
                        tokens[i] == pair_a and
                        tokens[i + 1] == pair_b):
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

            all_tokens.extend(tokens)

        return all_tokens

    def decode(self, tokens: list[str]) -> str:
        """Join tokens back into text."""
        text = ''.join(tokens)
        # Replace end-of-word markers with spaces
        text = text.replace('</w>', ' ')
        return text.rstrip()  # Remove trailing space

    @property
    def vocab(self) -> list[str]:
        return list(self._vocab)
```

**Testing:**
```python
# Test 1: Learn BPE merges
print("=" * 70)
print("TEST 1: Learn BPE Merges")
print("=" * 70)

corpus = "low lower newest widest low low lower newest newest newest widest"
merges = learn_bpe(corpus, num_merges=10)

print(f"Corpus: {corpus}")
print(f"\nLearned merges:")
for i, (a, b) in enumerate(merges):
    print(f"  {i+1}. '{a}' + '{b}' → '{a+b}'")

# Test 2: Encode and decode
print("\n" + "=" * 70)
print("TEST 2: Encode and Decode")
print("=" * 70)

tokenizer = BPETokenizer(merges)
text = "low newest widest"
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)

print(f"Input:   '{text}'")
print(f"Tokens:  {tokens}")
print(f"Decoded: '{decoded}'")
print(f"Roundtrip match: {text == decoded}")

# Test 3: Common words get fewer tokens
print("\n" + "=" * 70)
print("TEST 3: Frequent Words → Fewer Tokens")
print("=" * 70)

for word in ["low", "newest", "widest", "unknown"]:
    tokens = tokenizer.encode(word)
    # Remove </w> for display
    display_tokens = [t for t in tokens if t != '</w>']
    print(f"  '{word}' → {display_tokens} ({len(display_tokens)} tokens)")

# Test 4: Vocabulary size
print("\n" + "=" * 70)
print("TEST 4: Vocabulary")
print("=" * 70)
print(f"Vocab size: {len(tokenizer.vocab)}")
print(f"Vocab: {tokenizer.vocab[:20]}...")

# Test 5: More merges = shorter sequences
print("\n" + "=" * 70)
print("TEST 5: More Merges = Shorter Sequences")
print("=" * 70)

test_text = "newest lowest widest"
for n_merges in [0, 5, 10, 15]:
    m = learn_bpe(corpus, num_merges=n_merges)
    tok = BPETokenizer(m)
    tokens = tok.encode(test_text)
    print(f"  {n_merges:2d} merges → {len(tokens):2d} tokens: {tokens}")

# Test 6: Edge cases
print("\n" + "=" * 70)
print("TEST 6: Edge Cases")
print("=" * 70)

# Single character
tokens = tokenizer.encode("a")
print(f"Single char 'a': {tokens}")

# Already a known token
tokens = tokenizer.encode("low")
decoded = tokenizer.decode(tokens)
print(f"Known word 'low': tokens={tokens}, decoded='{decoded}', match={decoded == 'low'}")

# Multiple spaces become multiple words
tokens = tokenizer.encode("low low")
decoded = tokenizer.decode(tokens)
print(f"'low low': tokens={tokens}, decoded='{decoded}', match={decoded == 'low low'}")
```

**Common mistakes:**
1. Not using end-of-word markers (can't distinguish "lower" from "low" + "er" across word boundaries)
2. Applying merges in wrong order (must follow the learned order)
3. Greedy matching: BPE applies merges left to right, not finding the globally optimal segmentation
4. Forgetting to track word frequencies (counting pair frequency must be weighted by word frequency)
5. Not handling unknown characters at encoding time

## Follow-up Questions
- Why does BPE use an end-of-word marker?
- How does tokenization affect model security? (Token boundary attacks, prompt injection)
- What is the difference between BPE and WordPiece (used by BERT)?
- Why do modern models use byte-level BPE (like GPT-2)?
- How does vocabulary size affect model performance?
- What is SentencePiece and how does it differ from BPE?
