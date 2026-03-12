# Implement BPE Tokenizer
# ====================================================================
#
# Implement Byte-Pair Encoding (BPE) tokenization — the algorithm used by GPT, LLaMA, and most modern LLMs.
# 
# BPE starts with individual characters and iteratively merges the most frequent adjacent pair into a new token. This builds a vocabulary that balances between character-level (flexible, long sequences) and word-level (efficient, fixed vocab) tokenization.
# 
# Your implementation should include:
# 1. **`learn_bpe`**: Learn merge rules from a training corpus
# 2. **`BPETokenizer`**: Tokenizer class that encodes and decodes text
# 
# **Function signature:**
#
# ====================================================================

from collections import defaultdict
def learn_bpe(corpus: str, num_merges: int) -> list[tuple[str, str]]:
    """
    Learn BPE merge rules from a training corpus.

    Starting from individual characters, repeatedly find the most frequent
    adjacent pair and merge them into a new token.

    Args:
        corpus: training text
        num_merges: number of merge operations to learn
    Returns:
        merges: ordered list of (token_a, token_b) pairs to merge
    """
    merges = []
    corpus = [c for c in corpus]            
    for nm in range(num_merges):
        counts = defaultdict(int)
        for i in range(len(corpus)-1):
            counts[(corpus[i], corpus[i+1])] += 1
        # get the key whose value is max in counts        
        max_count = 0
        max_pair = None
        for k in counts.keys():
            if counts[k] > max_count:
                max_count = counts[k]
                max_pair = k
        merges.append(max_pair)
        # regenerate corpus
        new_corpus = []
        skip_next = False
        for i, c in enumerate(corpus):
            if skip_next: # skip this one since we combined it with the previous one 
                skip_next = False
                continue
            # last item, not matching max_pair[0], not matchin max_pair[1]
            if i == len(corpus) - 1 or c != max_pair[0] or corpus[i+1] != max_pair[1]:
                new_corpus.append(c)
            else: # add the combined corpus[i] and corpus[i+1]
                new_corpus.append(c + corpus[i+1])
                skip_next = True
        corpus = new_corpus
    return merges
        
            
class BPETokenizer:
    """
    Tokenizer using Byte-Pair Encoding.
    """
    def __init__(self, merges: list[tuple[str, str]]) -> None:
        """
        Args:
            merges: ordered list of merge rules from learn_bpe()
        """
        self.merges = merges

    def encode(self, text: str) -> list[str]:
        """
        Encode text into BPE tokens.

        Args:
            text: input string
        Returns:
            tokens: list of token strings
        """        
        tokens = [c for c in text]
        for m in self.merges:                
            merged_tokens = []        
            c1, c2 = m
            skip_next = False
            for i in range(len(tokens)):
                if skip_next:
                    skip_next = False
                    continue
                if i == len(tokens) - 1 or tokens[i] != c1 or tokens[i+1] != c2:
                    merged_tokens.append(tokens[i])
                else:
                    merged_tokens.append(c1+c2)
                    skip_next=True       
            tokens = merged_tokens
        return tokens 
            

    def decode(self, tokens: list[str]) -> str:
        """
        Decode BPE tokens back to text.

        Args:
            tokens: list of token strings
        Returns:
            text: decoded string
        """
        return ''.join(tokens)

    @property
    def vocab(self) -> list[str]:
        """Return the full vocabulary (base chars + merged tokens)."""                

