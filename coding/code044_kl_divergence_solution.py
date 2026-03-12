# Implement KL Divergence for RLHF
# ====================================================================
#
# Implement KL divergence computations used throughout RLHF and alignment pipelines.
# 
# KL divergence is the core regularizer in virtually every alignment algorithm:
# - **RLHF/PPO**: KL penalty `beta * KL(pi || pi_ref)` added to reward to prevent reward hacking
# - **DPO**: The KL constraint between policy and reference is baked into the loss derivation
# - **Constitutional AI**: Comparing model distributions over safe vs. unsafe completions
# 
# Your implementation should include:
# 1. **`kl_divergence`**: KL(P || Q) from logits — the "forward KL" used in most RLHF setups
# 2. **`token_level_kl`**: Per-token KL between policy and reference (used as a per-step penalty in PPO)
# 3. **`sequence_kl`**: Total KL over a full sequence — the quantity being bounded in RLHF
# 4. **`adaptive_kl_penalty`**: Adaptive controller that adjusts beta to hit a target KL
# 
# **Function signature:**
#
# ====================================================================

def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute KL(P || Q) from unnormalized logits.

    KL(P || Q) = sum_x P(x) * log(P(x) / Q(x))
               = sum_x P(x) * (log P(x) - log Q(x))

    Args:
        p_logits: logits for distribution P, shape (batch, vocab_size)
        q_logits: logits for distribution Q, shape (batch, vocab_size)
    Returns:
        kl: KL divergence per batch element, shape (batch,)
    """
    
    pass

def token_level_kl(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token KL(policy || ref) for a sequence.

    This is used in PPO-based RLHF as a per-token reward penalty:
    r_t = reward_t - beta * KL(policy(·|s_t) || ref(·|s_t))

    Args:
        policy_logits: logits from policy, shape (batch, seq_len, vocab_size)
        ref_logits: logits from reference model, shape (batch, seq_len, vocab_size)
    Returns:
        token_kl: per-token KL, shape (batch, seq_len)
    """
    pass

def sequence_kl(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Total KL divergence over a sequence (sum of per-token KLs).

    This is the quantity bounded in RLHF: E[KL(pi || pi_ref)] <= delta.

    Args:
        policy_logits: shape (batch, seq_len, vocab_size)
        ref_logits: shape (batch, seq_len, vocab_size)
        mask: optional bool/float mask, shape (batch, seq_len). True = include.
    Returns:
        seq_kl: shape (batch,)
    """
    pass

def adaptive_kl_penalty(
    kl_actual: float,
    kl_target: float,
    beta: float,
    kl_alpha: float = 0.1,
) -> float:
    """
    Adaptive KL controller: adjust beta to drive KL toward kl_target.

    From Anthropic's original RLHF paper (Ziegler et al. 2019):
    - If KL > 1.5 * target: increase beta (penalize harder)
    - If KL < target / 1.5: decrease beta (penalize softer)
    - Otherwise: keep beta

    Args:
        kl_actual: observed KL divergence this batch
        kl_target: desired KL divergence
        beta: current KL penalty coefficient
        kl_alpha: learning rate for beta update
    Returns:
        new_beta: updated KL coefficient
    """
    pass

