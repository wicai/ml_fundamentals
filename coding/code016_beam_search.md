# Implement Beam Search

**Category:** coding
**Difficulty:** 4
**Tags:** coding, inference, generation, search

## Question
Implement beam search for sequence generation.

Your implementation should:
- Maintain top-k candidates at each step
- Track cumulative log probabilities
- Support different beam widths
- Handle end-of-sequence tokens

**Function signature:**
```python
def beam_search(model, start_token, max_length, beam_width, vocab_size, eos_token=None):
    """
    Generate sequence using beam search.

    Args:
        model: function that takes (seq) and returns logits for next token
        start_token: initial token id
        max_length: maximum sequence length
        beam_width: number of beams to keep
        vocab_size: size of vocabulary
        eos_token: end-of-sequence token (optional)
    Returns:
        best_sequence: list of token ids
        best_score: log probability of best sequence
    """
    pass
```

## Answer

**Key concepts:**
1. Maintain beam_width candidates
2. Expand each candidate with all possible next tokens
3. Keep top beam_width by cumulative log probability
4. Stop when EOS token is generated or max_length reached

**Reference implementation:**
```python
import torch
import torch.nn.functional as F

def beam_search(model, start_token, max_length, beam_width, vocab_size, eos_token=None):
    """
    Simplified beam search implementation.
    """
    device = next(model.parameters()).device

    # Initialize with start token
    # Each beam: (sequence, cumulative_log_prob)
    beams = [([start_token], 0.0)]

    for step in range(max_length):
        all_candidates = []

        for seq, score in beams:
            # Skip if sequence already ended
            if eos_token is not None and seq[-1] == eos_token:
                all_candidates.append((seq, score))
                continue

            # Get model predictions for next token
            input_seq = torch.tensor([seq], device=device)
            with torch.no_grad():
                logits = model(input_seq)  # (1, seq_len, vocab_size)
                logits = logits[0, -1, :]  # Get last token's logits

            # Get log probabilities
            log_probs = F.log_softmax(logits, dim=-1)

            # Get top beam_width candidates
            top_log_probs, top_indices = torch.topk(log_probs, beam_width)

            # Add to candidates
            for i in range(beam_width):
                token = top_indices[i].item()
                token_log_prob = top_log_probs[i].item()

                new_seq = seq + [token]
                new_score = score + token_log_prob

                all_candidates.append((new_seq, new_score))

        # Keep top beam_width candidates by score
        beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        # Early stopping: if all beams have EOS
        if eos_token is not None and all(seq[-1] == eos_token for seq, _ in beams):
            break

    # Return best beam
    best_sequence, best_score = beams[0]
    return best_sequence, best_score

# More complete implementation with length normalization
def beam_search_advanced(
    model,
    start_token,
    max_length,
    beam_width,
    vocab_size,
    eos_token=None,
    length_penalty=1.0,
    early_stopping=True
):
    """
    Beam search with length normalization and early stopping.

    Args:
        length_penalty: divide score by (length ** length_penalty)
                       1.0 = no penalty, >1.0 = favor longer sequences
    """
    device = next(model.parameters()).device

    # Active beams: (sequence, cumulative_score, is_finished)
    beams = [([start_token], 0.0, False)]
    finished_beams = []

    for step in range(max_length):
        all_candidates = []

        for seq, score, is_finished in beams:
            if is_finished:
                all_candidates.append((seq, score, True))
                continue

            # Get predictions
            input_seq = torch.tensor([seq], device=device)
            with torch.no_grad():
                logits = model(input_seq)[0, -1, :]

            log_probs = F.log_softmax(logits, dim=-1)
            top_log_probs, top_indices = torch.topk(log_probs, beam_width)

            for i in range(beam_width):
                token = top_indices[i].item()
                token_log_prob = top_log_probs[i].item()

                new_seq = seq + [token]
                new_score = score + token_log_prob

                # Check if finished
                is_eos = (eos_token is not None and token == eos_token)

                all_candidates.append((new_seq, new_score, is_eos))

        # Separate finished and unfinished beams
        finished = [(seq, score) for seq, score, is_fin in all_candidates if is_fin]
        unfinished = [(seq, score, is_fin) for seq, score, is_fin in all_candidates if not is_fin]

        # Add finished beams to finished list
        finished_beams.extend(finished)

        # Keep top beam_width unfinished beams
        beams = sorted(unfinished, key=lambda x: x[1], reverse=True)[:beam_width]

        # Early stopping: if we have enough finished beams
        if early_stopping and len(finished_beams) >= beam_width:
            break

        # If no more active beams, stop
        if len(beams) == 0:
            break

    # Combine finished and remaining beams
    all_beams = finished_beams + [(seq, score) for seq, score, _ in beams]

    # Apply length normalization
    normalized_beams = []
    for seq, score in all_beams:
        normalized_score = score / (len(seq) ** length_penalty)
        normalized_beams.append((seq, normalized_score))

    # Return best
    best_sequence, best_score = max(normalized_beams, key=lambda x: x[1])

    return best_sequence, best_score
```

**Testing:**
```python
# Mock model for testing
class MockModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x):
        # Return random logits
        batch_size, seq_len = x.shape
        return torch.randn(batch_size, seq_len, self.vocab_size)

# Test
vocab_size = 100
model = MockModel(vocab_size)
start_token = 0
eos_token = 1

# Beam search
sequence, score = beam_search(
    model,
    start_token=start_token,
    max_length=20,
    beam_width=5,
    vocab_size=vocab_size,
    eos_token=eos_token
)

print(f"Generated sequence: {sequence}")
print(f"Score: {score:.4f}")
print(f"Length: {len(sequence)}")

# Compare different beam widths
for beam_width in [1, 3, 5, 10]:
    seq, score = beam_search(
        model, start_token, 20, beam_width, vocab_size, eos_token
    )
    print(f"Beam width {beam_width}: length={len(seq)}, score={score:.4f}")

# Test with length penalty
seq_no_penalty, score_no_penalty = beam_search_advanced(
    model, start_token, 20, 5, vocab_size, eos_token, length_penalty=0.0
)

seq_with_penalty, score_with_penalty = beam_search_advanced(
    model, start_token, 20, 5, vocab_size, eos_token, length_penalty=0.6
)

print(f"\nNo penalty: length={len(seq_no_penalty)}")
print(f"With penalty: length={len(seq_with_penalty)}")
```

**Common mistakes:**
1. ❌ Not tracking cumulative log probabilities
2. ❌ Expanding all vocab tokens instead of top-k
3. ❌ Forgetting to handle finished sequences
4. ❌ Not normalizing by length (favors shorter sequences)

## Follow-up Questions
- What's the time complexity of beam search?
- How does beam width affect quality vs speed?
- When would you use beam search vs sampling?
