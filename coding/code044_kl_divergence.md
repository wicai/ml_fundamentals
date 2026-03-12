# Implement KL Divergence for RLHF

**Category:** coding
**Difficulty:** 3
**Tags:** coding, rlhf, kl-divergence, alignment, safety

## Question

Implement KL divergence computations used throughout RLHF and alignment pipelines.

KL divergence is the core regularizer in virtually every alignment algorithm:
- **RLHF/PPO**: KL penalty `beta * KL(pi || pi_ref)` added to reward to prevent reward hacking
- **DPO**: The KL constraint between policy and reference is baked into the loss derivation
- **Constitutional AI**: Comparing model distributions over safe vs. unsafe completions

Your implementation should include:
1. **`kl_divergence`**: KL(P || Q) from logits — the "forward KL" used in most RLHF setups
2. **`token_level_kl`**: Per-token KL between policy and reference (used as a per-step penalty in PPO)
3. **`sequence_kl`**: Total KL over a full sequence — the quantity being bounded in RLHF
4. **`adaptive_kl_penalty`**: Adaptive controller that adjusts beta to hit a target KL

**Function signature:**
```python
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
```

## Answer

**Key concepts:**
1. KL(P || Q) is always >= 0, equals 0 iff P = Q (Gibbs' inequality)
2. KL is **asymmetric**: KL(P || Q) != KL(Q || P)
3. In RLHF, we always compute KL(policy || reference), not the reverse — this penalizes the policy for diverging from reference
4. Forward KL (P || Q) is "mode-seeking": P tries to cover all modes of Q
5. The sum of per-token KLs equals the sequence-level KL for autoregressive models: `KL(pi(y|x) || ref(y|x)) = sum_t KL(pi(a_t|...) || ref(a_t|...))`
6. Use `F.kl_div` carefully — it expects log-probabilities for input and probabilities for target, and uses a non-standard sign convention

**Reference implementation:**
```python
import torch
import torch.nn.functional as F

def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(P || Q) from logits."""
    # Convert logits to log-probabilities
    log_p = F.log_softmax(p_logits, dim=-1)  # log P(x)
    log_q = F.log_softmax(q_logits, dim=-1)  # log Q(x)
    p = log_p.exp()                           # P(x)

    # KL(P || Q) = sum_x P(x) * (log P(x) - log Q(x))
    # Shape: (batch,)
    return (p * (log_p - log_q)).sum(dim=-1)

def token_level_kl(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
) -> torch.Tensor:
    """Per-token KL(policy || ref), shape (batch, seq_len)."""
    log_policy = F.log_softmax(policy_logits, dim=-1)  # (batch, seq, vocab)
    log_ref = F.log_softmax(ref_logits, dim=-1)
    policy = log_policy.exp()

    # Sum over vocab at each position
    return (policy * (log_policy - log_ref)).sum(dim=-1)

def sequence_kl(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Total KL over a sequence."""
    token_kl = token_level_kl(policy_logits, ref_logits)  # (batch, seq_len)

    if mask is not None:
        return (token_kl * mask).sum(dim=-1)
    return token_kl.sum(dim=-1)

def adaptive_kl_penalty(
    kl_actual: float,
    kl_target: float,
    beta: float,
    kl_alpha: float = 0.1,
) -> float:
    """Adjust beta to drive KL toward target."""
    if kl_actual > 1.5 * kl_target:
        # KL too high: increase penalty
        return beta * (1 + kl_alpha)
    elif kl_actual < kl_target / 1.5:
        # KL too low: decrease penalty (policy too conservative)
        return beta * (1 - kl_alpha)
    else:
        return beta
```

**Testing:**
```python
import torch
import torch.nn.functional as F

torch.manual_seed(1)

# Test 1: KL properties
print("=" * 70)
print("TEST 1: KL Divergence Properties")
print("=" * 70)

# KL(P || P) = 0
p_logits = torch.randn(4, 100)
kl_self = kl_divergence(p_logits, p_logits)
print(f"KL(P || P) = {kl_self.mean().item():.2e}  (should be ~0)")

# KL >= 0
q_logits = torch.randn(4, 100)
kl = kl_divergence(p_logits, q_logits)
print(f"KL >= 0: {(kl >= 0).all().item()}")
print(f"Sample KL values: {kl.tolist()}")

# KL is asymmetric
kl_pq = kl_divergence(p_logits, q_logits)
kl_qp = kl_divergence(q_logits, p_logits)
print(f"KL(P||Q) != KL(Q||P): {not torch.allclose(kl_pq, kl_qp)}")
print(f"KL(P||Q): {kl_pq.mean().item():.4f}")
print(f"KL(Q||P): {kl_qp.mean().item():.4f}")

# Test 2: Token-level KL
print("\n" + "=" * 70)
print("TEST 2: Token-Level KL")
print("=" * 70)

batch, seq_len, vocab = 2, 10, 50
policy_logits = torch.randn(batch, seq_len, vocab)
ref_logits = torch.randn(batch, seq_len, vocab)

token_kl = token_level_kl(policy_logits, ref_logits)
print(f"Token KL shape: {token_kl.shape}  (should be ({batch}, {seq_len}))")
print(f"All token KL >= 0: {(token_kl >= 0).all().item()}")

# Test 3: Sequence KL = sum of token KLs
print("\n" + "=" * 70)
print("TEST 3: Sequence KL = Sum of Token KLs")
print("=" * 70)

seq_kl = sequence_kl(policy_logits, ref_logits)
manual_seq_kl = token_level_kl(policy_logits, ref_logits).sum(dim=-1)
print(f"Sequence KL:      {seq_kl.tolist()}")
print(f"Manual sum:       {manual_seq_kl.tolist()}")
print(f"Match: {torch.allclose(seq_kl, manual_seq_kl)}")

# Test 4: Masked sequence KL (only score completion tokens)
print("\n" + "=" * 70)
print("TEST 4: Masked Sequence KL")
print("=" * 70)

mask = torch.zeros(batch, seq_len)
mask[:, 5:] = 1.0  # Only score last 5 tokens
masked_seq_kl = sequence_kl(policy_logits, ref_logits, mask)
manual_masked = token_level_kl(policy_logits, ref_logits)[:, 5:].sum(dim=-1)
print(f"Masked KL:  {masked_seq_kl.tolist()}")
print(f"Manual:     {manual_masked.tolist()}")
print(f"Match: {torch.allclose(masked_seq_kl, manual_masked)}")

# Test 5: Adaptive KL controller
print("\n" + "=" * 70)
print("TEST 5: Adaptive KL Controller")
print("=" * 70)

beta = 0.1
target = 0.05

# KL too high: should increase beta
beta_new = adaptive_kl_penalty(kl_actual=0.2, kl_target=target, beta=beta)
print(f"KL too high (0.2 > 1.5 * {target}): beta {beta} -> {beta_new:.4f} (should increase)")

# KL too low: should decrease beta
beta_new = adaptive_kl_penalty(kl_actual=0.01, kl_target=target, beta=beta)
print(f"KL too low (0.01 < {target} / 1.5): beta {beta} -> {beta_new:.4f} (should decrease)")

# KL in range: beta unchanged
beta_new = adaptive_kl_penalty(kl_actual=0.05, kl_target=target, beta=beta)
print(f"KL in range (0.05 ~ {target}): beta {beta} -> {beta_new:.4f} (should stay same)")

# Test 6: Verify against F.kl_div
print("\n" + "=" * 70)
print("TEST 6: Verify Against F.kl_div")
print("=" * 70)
p_logits = torch.randn(3, 50)
q_logits = torch.randn(3, 50)
our_kl = kl_divergence(p_logits, q_logits)

# F.kl_div(log_input, target) computes target * (log_target - log_input)
# i.e., KL(target || input), so input = Q, target = P
log_p = F.log_softmax(p_logits, dim=-1)
log_q = F.log_softmax(q_logits, dim=-1)
p = log_p.exp()
torch_kl = F.kl_div(log_q, p, reduction='none').sum(dim=-1)  # KL(P || Q)
print(f"Our KL:   {our_kl.tolist()}")
print(f"F.kl_div: {torch_kl.tolist()}")
print(f"Match: {torch.allclose(our_kl, torch_kl, atol=1e-5)}")
```

**Common mistakes:**
1. Confusing KL(P || Q) vs KL(Q || P) — in RLHF it's always KL(policy || reference)
2. Using `F.kl_div` incorrectly: it takes `(log_input, target)` and computes `target * (log_target - log_input)` — the argument order is backwards from what you'd expect
3. Forgetting that KL sums over the vocabulary at each token position
4. Using raw probabilities instead of log-probabilities in intermediate steps (numerical instability)
5. Treating KL as symmetric — it's not, and the choice of direction matters

## Follow-up Questions
- Why do we compute KL(policy || reference) and not KL(reference || policy) in RLHF?
- What happens when beta → 0? When beta → infinity? What's a typical value?
- Why does high KL between policy and reference suggest reward hacking?
- How does the KL term relate to the "implicit reward" in DPO?
- What is the relationship between KL divergence and cross-entropy?
- Why is adaptive KL control useful compared to a fixed beta?
