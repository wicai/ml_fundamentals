# Implement PPO Clipped Surrogate Objective

**Category:** coding
**Difficulty:** 4
**Tags:** coding, rlhf, ppo, policy-gradient, alignment, safety

## Question

Implement the PPO (Proximal Policy Optimization) clipped surrogate objective used in RLHF. This is the policy gradient loss at the heart of Anthropic's early Claude training, OpenAI's InstructGPT, and many other RLHF systems.

**Background:** Standard policy gradient (REINFORCE) has high variance and takes large, destabilizing steps. PPO constrains the policy update by clipping the probability ratio `r_t = pi(a|s) / pi_old(a|s)`, preventing any single update from moving the policy too far.

Your implementation should include:
1. **`compute_policy_ratio`**: Ratio of new vs. old policy probabilities
2. **`ppo_clip_loss`**: The clipped surrogate objective — the core of PPO
3. **`gae`**: Generalized Advantage Estimation — how PPO estimates which actions were good
4. **`ppo_epoch`**: Run multiple gradient steps on the same batch of experience (the "proximal" in PPO)

**Function signature:**
```python
def compute_policy_ratio(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the probability ratio r_t = pi_new(a|s) / pi_old(a|s).

    Computed in log space for numerical stability.

    Args:
        log_probs_new: log pi(a_t|s_t) under new policy, shape (batch,)
        log_probs_old: log pi(a_t|s_t) under old policy, shape (batch,)
    Returns:
        ratio: pi_new / pi_old, shape (batch,). Should be ~1.0 at start of training.
    """
    pass

def ppo_clip_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    PPO clipped surrogate objective.

    L_CLIP = -E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]

    The clip prevents too-large policy updates:
    - If A > 0 (good action): don't let ratio exceed 1 + eps (don't over-exploit)
    - If A < 0 (bad action): don't let ratio go below 1 - eps (don't over-punish)

    Args:
        log_probs_new: log pi(a|s) under new policy, shape (batch,)
        log_probs_old: log pi(a|s) under old policy (detached), shape (batch,)
        advantages: advantage estimates A_t, shape (batch,). Normalized recommended.
        clip_eps: clipping threshold (default 0.2 per original PPO paper)
    Returns:
        loss: scalar (negative because we minimize loss = maximize objective)
        info: dict with 'clip_fraction', 'approx_kl', 'policy_ratio_mean'
    """
    pass

def gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation (GAE-lambda).

    Computes advantages A_t = sum_{l=0}^{inf} (gamma * lam)^l * delta_{t+l}
    where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t) is the TD error.

    Args:
        rewards: shape (batch, seq_len)
        values: shape (batch, seq_len + 1) — includes bootstrap value V(s_T)
        dones: shape (batch, seq_len) — 1.0 if episode ended, 0.0 otherwise
        gamma: discount factor (0.99 typical)
        lam: GAE lambda — 0 = TD(0), 1 = Monte Carlo (0.95 typical)
    Returns:
        advantages: shape (batch, seq_len)
        returns: shape (batch, seq_len) — targets for value function (advantages + values)
    """
    pass

def ppo_epoch(
    policy_model: torch.nn.Module,
    value_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    input_ids: torch.Tensor,
    n_epochs: int = 4,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> dict[str, float]:
    """
    Run n_epochs of PPO updates on a fixed batch of experience.

    Total loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    Args:
        policy_model: outputs logits (batch, seq_len, vocab_size)
        value_model: outputs scalar values (batch,)
        optimizer: shared optimizer for both models
        log_probs_old: log probs at time of data collection, shape (batch,)
        advantages: GAE advantages, shape (batch,)
        returns: value targets, shape (batch,)
        input_ids: token IDs, shape (batch, seq_len)
        n_epochs: how many gradient steps on this batch (PPO re-uses data)
        clip_eps: PPO clip threshold
        vf_coef: value function loss coefficient
        ent_coef: entropy bonus coefficient (encourages exploration)
    Returns:
        dict with avg 'policy_loss', 'value_loss', 'entropy', 'clip_fraction', 'approx_kl'
    """
    pass
```

## Answer

**Key concepts:**
1. `r_t = pi_new / pi_old` — if ratio > 1, new policy assigns higher probability to this action
2. The clip `min(r * A, clip(r, 1-eps, 1+eps) * A)` prevents the policy from taking overly large steps in either direction
3. **Advantages should be normalized** (zero mean, unit variance) before computing the loss — this is a critical implementation detail
4. PPO re-uses each batch of data for `n_epochs` gradient steps — this is computationally efficient but only safe because of the clipping
5. The **entropy bonus** prevents premature convergence to deterministic policies — important for safety/diversity
6. GAE with `lam < 1` trades variance for bias: lower lambda = lower variance but biased estimates

**Reference implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_policy_ratio(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
) -> torch.Tensor:
    """r_t = pi_new / pi_old = exp(log_pi_new - log_pi_old)."""
    return torch.exp(log_probs_new - log_probs_old)

def ppo_clip_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PPO clipped surrogate objective."""
    # Normalize advantages (critical for training stability)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    ratio = compute_policy_ratio(log_probs_new, log_probs_old)

    # Unclipped: standard policy gradient term
    unclipped = ratio * advantages

    # Clipped: bound how much ratio can deviate from 1
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    clipped = clipped_ratio * advantages

    # Take the min — this is the pessimistic (safe) bound
    policy_loss = -torch.min(unclipped, clipped).mean()

    # Diagnostics
    clip_fraction = ((ratio - 1).abs() > clip_eps).float().mean().item()
    approx_kl = (log_probs_old - log_probs_new).mean().item()  # approx KL(old || new)

    info = {
        'clip_fraction': clip_fraction,
        'approx_kl': approx_kl,
        'policy_ratio_mean': ratio.mean().item(),
    }

    return policy_loss, info

def gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation."""
    batch, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)

    # Bootstrap from final value
    last_gae = 0.0

    # Iterate backwards through time steps
    for t in reversed(range(seq_len)):
        # 0 if done (episode ended), so next value is 0
        not_done = 1.0 - dones[:, t]

        # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards[:, t] + gamma * values[:, t + 1] * not_done - values[:, t]

        # GAE: accumulate discounted TD errors
        last_gae = delta + gamma * lam * not_done * last_gae
        advantages[:, t] = last_gae

    returns = advantages + values[:, :-1]
    return advantages, returns

def _get_log_probs(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Get sequence-level log-probability (sum over tokens) from a language model.
    """
    logits = model(input_ids)  # (batch, seq_len, vocab)
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lps = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_lps.sum(dim=-1)  # (batch,)

def _get_entropy(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Mean token-level entropy of the policy."""
    logits = model(input_ids)[:, :-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    # H(x) = -sum P(x) log P(x), averaged over tokens and batch
    return -(probs * log_probs).sum(dim=-1).mean()

def ppo_epoch(
    policy_model: nn.Module,
    value_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    input_ids: torch.Tensor,
    n_epochs: int = 4,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> dict[str, float]:
    """Run PPO update for n_epochs on a fixed batch."""
    metrics = {k: 0.0 for k in ['policy_loss', 'value_loss', 'entropy', 'clip_fraction', 'approx_kl']}

    for _ in range(n_epochs):
        policy_model.train()
        value_model.train()

        # New log-probs under current policy
        log_probs_new = _get_log_probs(policy_model, input_ids)

        # Policy loss
        policy_loss, info = ppo_clip_loss(log_probs_new, log_probs_old.detach(), advantages, clip_eps)

        # Value function loss (MSE between predicted values and returns)
        values_pred = value_model(input_ids)  # (batch,)
        value_loss = F.mse_loss(values_pred, returns)

        # Entropy bonus (maximize to encourage exploration)
        entropy = _get_entropy(policy_model, input_ids)

        # Combined loss
        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        metrics['policy_loss'] += policy_loss.item()
        metrics['value_loss'] += value_loss.item()
        metrics['entropy'] += entropy.item()
        metrics['clip_fraction'] += info['clip_fraction']
        metrics['approx_kl'] += info['approx_kl']

    # Average over epochs
    return {k: v / n_epochs for k, v in metrics.items()}
```

**Testing:**
```python
import torch
import torch.nn as nn

torch.manual_seed(1)

# Test 1: Policy ratio basics
print("=" * 70)
print("TEST 1: Policy Ratio")
print("=" * 70)

log_probs_old = torch.tensor([-2.0, -1.5, -3.0, -1.0])
log_probs_new = torch.tensor([-2.0, -1.5, -3.0, -1.0])  # Same policy
ratio = compute_policy_ratio(log_probs_new, log_probs_old)
print(f"Ratio (same policy): {ratio.tolist()} (should all be 1.0)")

log_probs_new_higher = log_probs_old + 0.5  # New policy more likely
ratio_higher = compute_policy_ratio(log_probs_new_higher, log_probs_old)
print(f"Ratio (new prob higher): {ratio_higher.tolist()} (should all be ~1.65)")

# Test 2: PPO clip loss
print("\n" + "=" * 70)
print("TEST 2: PPO Clip Loss")
print("=" * 70)

log_probs_old = torch.randn(8)
advantages = torch.randn(8)

# When new == old (start of training), loss should be pure policy gradient
log_probs_new = log_probs_old.clone()
loss_same, info = ppo_clip_loss(log_probs_new, log_probs_old, advantages, clip_eps=0.2)
print(f"Clip fraction (same policy): {info['clip_fraction']:.2f} (should be 0)")

# When new policy diverges significantly, clipping kicks in
log_probs_diverged = log_probs_old + 5.0  # Very different policy
loss_div, info_div = ppo_clip_loss(log_probs_diverged, log_probs_old, advantages, clip_eps=0.2)
print(f"Clip fraction (diverged): {info_div['clip_fraction']:.2f} (should be 1.0)")
print(f"Approx KL: {info_div['approx_kl']:.4f}")

# Test 3: Clipping effect — loss is bounded
print("\n" + "=" * 70)
print("TEST 3: Clipping Bounds Policy Updates")
print("=" * 70)

# Positive advantage: we want to increase log_prob, but not too much
advantages_pos = torch.ones(4) * 2.0  # All positive, all same magnitude
for delta in [0.0, 0.1, 0.2, 1.0, 5.0]:
    lp_new = log_probs_old[:4] + delta
    loss, info = ppo_clip_loss(lp_new, log_probs_old[:4], advantages_pos, clip_eps=0.2)
    print(f"  delta={delta:.1f}: loss={-loss.item():.4f}, clip_frac={info['clip_fraction']:.2f}")

print("Note: objective stops increasing once ratio exceeds 1 + clip_eps")

# Test 4: GAE
print("\n" + "=" * 70)
print("TEST 4: Generalized Advantage Estimation")
print("=" * 70)

batch, seq_len = 2, 5
rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0],
                         [0.0, 0.0, 0.0, 0.0, -1.0]])
values = torch.zeros(batch, seq_len + 1)  # V(s) = 0 for simplicity
dones = torch.zeros(batch, seq_len)

advantages, returns = gae(rewards, values, dones, gamma=0.99, lam=0.95)
print(f"Advantages shape: {advantages.shape}")
print(f"Returns shape: {returns.shape}")
print(f"Sequence 1 advantages: {advantages[0].tolist()}")
print(f"Sequence 2 advantages: {advantages[1].tolist()}")
print(f"Advantage at final reward step > 0 (positive reward): {advantages[0, -1].item() > 0}")
print(f"Advantage at final reward step < 0 (negative reward): {advantages[1, -1].item() < 0}")

# Test 5: Lambda effect on GAE
print("\n" + "=" * 70)
print("TEST 5: Lambda Effect on GAE")
print("=" * 70)

rewards_simple = torch.tensor([[0.0, 0.0, 0.0, 0.0, 10.0]])
values_simple = torch.zeros(1, 6)
dones_simple = torch.zeros(1, 5)

print("Reward only at last step. Lambda controls how far back credit is assigned:")
for lam in [0.0, 0.5, 0.95, 1.0]:
    adv, _ = gae(rewards_simple, values_simple, dones_simple, gamma=0.99, lam=lam)
    print(f"  lam={lam}: advantages = {[f'{a:.3f}' for a in adv[0].tolist()]}")
print("lam=0: only immediate credit (TD(0)). lam=1: full Monte Carlo return.")
```

**Common mistakes:**
1. Forgetting to normalize advantages before computing the loss — training becomes very unstable
2. Not detaching `log_probs_old` — they should be treated as fixed constants from the old policy
3. Getting the clip direction wrong: we clip the *ratio*, not the log-prob difference
4. Forgetting the negative sign — we **maximize** the objective but PyTorch optimizers **minimize**
5. Off-by-one in GAE: `values` should have shape `(batch, seq_len + 1)` to include the bootstrap value
6. Using `reduction='mean'` in value loss when you want per-token weighting

## Follow-up Questions
- Why does PPO clip the ratio rather than bounding the KL directly (like TRPO)?
- What is the "clip fraction" metric and what does a high value indicate?
- How does the entropy bonus relate to exploration vs. exploitation in RLHF?
- What does approx_kl estimate, and why is it useful to monitor during training?
- What happens if you run too many PPO epochs on the same batch? (overfitting to old experience)
- How does PPO-based RLHF differ from DPO in practice? When would you prefer each?
- What is the "value function" doing in RLHF, and why is it needed for GAE?
