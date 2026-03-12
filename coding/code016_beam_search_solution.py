# Implement Beam Search
# ====================================================================
#
# Implement beam search for sequence generation.
# 
# Your implementation should:
# - Maintain top-k candidates at each step
# - Track cumulative log probabilities
# - Support different beam widths
# - Handle end-of-sequence tokens
# 
# **Function signature:**
#
# ====================================================================

from typing import Callable, Optional

import torch


def beam_search(
    model: Callable[[list[int]], torch.Tensor],
    start_token: int,
    max_length: int,
    beam_width: int,
    vocab_size: int,
    eos_token: Optional[int] = None,
) -> tuple[list[int], float]:
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

