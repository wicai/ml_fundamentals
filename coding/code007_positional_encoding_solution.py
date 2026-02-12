# Implement Sinusoidal Positional Encoding
# ====================================================================
#
# Implement sinusoidal positional encoding as used in the original Transformer paper.
# 
# Your implementation should:
# - Generate position encodings for any sequence length
# - Use sine for even dimensions, cosine for odd dimensions
# - Support different model dimensions
# - Be deterministic (not learned)
# 
# **Function signature:**
#
# ====================================================================

import torch

def get_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """
    Generate sinusoidal positional encodings.

    Args:
        seq_len: sequence length
        d_model: model dimension (embedding size)
    Returns:
        positional encoding tensor of shape (seq_len, d_model)
    """
    pass

