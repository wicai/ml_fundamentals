# Implement DPO (Direct Preference Optimization) Loss

**Category:** coding
**Difficulty:** 4
**Tags:** coding, dpo, alignment, safety, rlhf

## Question
Implement the DPO loss function. DPO skips training a separate reward model and directly optimizes the policy using preference data.

The key insight: the optimal policy under a KL-constrained reward maximization objective can be expressed as a closed-form function of the policy's own log-probabilities relative to a reference model.

Your implementation should include:
1. **`dpo_loss`**: The core DPO objective
2. **`compute_log_ratio`**: Helper to compute log(pi/ref) for a sequence
3. **`dpo_training_step`**: A single training step

**Function signature:**
```python
def compute_log_ratio(
    policy_chosen_logprobs: torch.Tensor,
    policy_rejected_logprobs: torch.Tensor,
    ref_chosen_logprobs: torch.Tensor,
    ref_rejected_logprobs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute log-probability ratios: log(pi(y|x) / ref(y|x)) for chosen and rejected.

    All inputs are *sequence-level* log-probs (already summed over tokens).

    Args:
        policy_chosen_logprobs: log pi(y_chosen | x), shape (batch_size,)
        policy_rejected_logprobs: log pi(y_rejected | x), shape (batch_size,)
        ref_chosen_logprobs: log ref(y_chosen | x), shape (batch_size,)
        ref_rejected_logprobs: log ref(y_rejected | x), shape (batch_size,)
    Returns:
        chosen_log_ratios: log(pi/ref) for chosen, shape (batch_size,)
        rejected_log_ratios: log(pi/ref) for rejected, shape (batch_size,)
    """
    pass

def dpo_loss(
    policy_chosen_logprobs: torch.Tensor,
    policy_rejected_logprobs: torch.Tensor,
    ref_chosen_logprobs: torch.Tensor,
    ref_rejected_logprobs: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute DPO loss.

    L_DPO = -E[log sigmoid(beta * (log(pi_chosen/ref_chosen) - log(pi_rejected/ref_rejected)))]

    Args:
        policy_chosen_logprobs: log pi(y_chosen | x), shape (batch_size,)
        policy_rejected_logprobs: log pi(y_rejected | x), shape (batch_size,)
        ref_chosen_logprobs: log ref(y_chosen | x), shape (batch_size,)
        ref_rejected_logprobs: log ref(y_rejected | x), shape (batch_size,)
        beta: temperature parameter controlling deviation from reference
    Returns:
        loss: scalar DPO loss
        chosen_rewards: implicit rewards for chosen, shape (batch_size,)
        rejected_rewards: implicit rewards for rejected, shape (batch_size,)
    """
    pass

def dpo_training_step(
    policy_model: nn.Module,
    ref_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    beta: float = 0.1,
) -> dict[str, float]:
    """
    One DPO training step.

    Args:
        policy_model: the model being trained
        ref_model: frozen reference model (usually the SFT model)
        optimizer: optimizer for policy_model
        chosen_ids: token IDs for chosen responses, shape (batch_size, seq_len)
        rejected_ids: token IDs for rejected responses, shape (batch_size, seq_len)
        beta: DPO temperature
    Returns:
        dict with 'loss', 'accuracy', 'chosen_reward', 'rejected_reward'
    """
    pass
```

## Answer

**Key concepts:**
1. DPO implicitly defines a reward: `r(x, y) = beta * log(pi(y|x) / ref(y|x))`
2. The loss has the same form as the reward model loss but uses log-ratios as rewards
3. beta controls how far the policy can deviate from the reference
4. The reference model is frozen — only the policy is updated
5. Higher beta = more conservative (stays closer to reference)

**Reference implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_log_ratio(
    policy_chosen_logprobs: torch.Tensor,
    policy_rejected_logprobs: torch.Tensor,
    ref_chosen_logprobs: torch.Tensor,
    ref_rejected_logprobs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log(pi/ref) for chosen and rejected."""
    chosen_log_ratios = policy_chosen_logprobs - ref_chosen_logprobs
    rejected_log_ratios = policy_rejected_logprobs - ref_rejected_logprobs
    return chosen_log_ratios, rejected_log_ratios

def dpo_loss(
    policy_chosen_logprobs: torch.Tensor,
    policy_rejected_logprobs: torch.Tensor,
    ref_chosen_logprobs: torch.Tensor,
    ref_rejected_logprobs: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    DPO loss = -log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))

    This is equivalent to the reward model loss where:
    implicit_reward = beta * log(pi(y|x) / ref(y|x))
    """
    chosen_log_ratios, rejected_log_ratios = compute_log_ratio(
        policy_chosen_logprobs, policy_rejected_logprobs,
        ref_chosen_logprobs, ref_rejected_logprobs,
    )

    # The implicit rewards
    chosen_rewards = beta * chosen_log_ratios
    rejected_rewards = beta * rejected_log_ratios

    # DPO loss: same as preference loss on implicit rewards
    logits = chosen_rewards - rejected_rewards  # = beta * (log_ratio_c - log_ratio_r)
    loss = -F.logsigmoid(logits).mean()

    return loss, chosen_rewards.detach(), rejected_rewards.detach()

def _get_sequence_logprobs(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Helper: get per-sequence log-probabilities from a language model.

    For each sequence, compute sum of log P(token_t | tokens_<t).
    """
    # Get logits from model
    logits = model(input_ids)  # (batch, seq_len, vocab_size)

    # Shift: predict next token from current
    # logits[:, :-1] predicts input_ids[:, 1:]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # Per-token log-probs
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Sum over sequence to get sequence-level log-prob
    return token_log_probs.sum(dim=-1)

def dpo_training_step(
    policy_model: nn.Module,
    ref_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    beta: float = 0.1,
) -> dict[str, float]:
    """One DPO training step."""
    policy_model.train()
    ref_model.eval()

    # Get policy log-probs (with gradients)
    policy_chosen_logprobs = _get_sequence_logprobs(policy_model, chosen_ids)
    policy_rejected_logprobs = _get_sequence_logprobs(policy_model, rejected_ids)

    # Get reference log-probs (no gradients)
    with torch.no_grad():
        ref_chosen_logprobs = _get_sequence_logprobs(ref_model, chosen_ids)
        ref_rejected_logprobs = _get_sequence_logprobs(ref_model, rejected_ids)

    # Compute loss
    loss, chosen_rewards, rejected_rewards = dpo_loss(
        policy_chosen_logprobs, policy_rejected_logprobs,
        ref_chosen_logprobs, ref_rejected_logprobs,
        beta=beta,
    )

    # Update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

    return {
        'loss': loss.item(),
        'accuracy': accuracy,
        'chosen_reward': chosen_rewards.mean().item(),
        'rejected_reward': rejected_rewards.mean().item(),
    }
```

**Testing:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# Test 1: DPO loss basic properties
print("=" * 70)
print("TEST 1: DPO Loss Properties")
print("=" * 70)

# When policy matches reference, log-ratios are 0, loss = log(2)
pi_c = torch.tensor([-5.0, -3.0])
pi_r = torch.tensor([-6.0, -4.0])
ref_c = torch.tensor([-5.0, -3.0])  # Same as policy
ref_r = torch.tensor([-6.0, -4.0])  # Same as policy

loss_equal, _, _ = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1)
print(f"Policy == Reference: loss = {loss_equal.item():.4f} (should be ~{torch.log(torch.tensor(2.0)).item():.4f})")

# When policy strongly prefers chosen over reference, loss should be low
pi_c_better = torch.tensor([-2.0, -1.0])  # Policy assigns higher prob to chosen
loss_good, c_rew, r_rew = dpo_loss(pi_c_better, pi_r, ref_c, ref_r, beta=0.1)
print(f"Policy prefers chosen: loss = {loss_good.item():.4f} (should be < {loss_equal.item():.4f})")
print(f"  chosen_reward > rejected_reward: {(c_rew > r_rew).all().item()}")

# Test 2: Beta effect
print("\n" + "=" * 70)
print("TEST 2: Beta Controls Conservatism")
print("=" * 70)

pi_c = torch.tensor([-3.0])
pi_r = torch.tensor([-5.0])
ref_c = torch.tensor([-4.0])
ref_r = torch.tensor([-4.0])

for beta in [0.01, 0.1, 0.5, 1.0]:
    loss, c_rew, r_rew = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=beta)
    print(f"beta={beta:.2f}: loss={loss.item():.4f}, reward_gap={((c_rew - r_rew).item()):.4f}")

# Test 3: Gradient direction
print("\n" + "=" * 70)
print("TEST 3: Gradient Direction")
print("=" * 70)

pi_c = torch.tensor([-5.0], requires_grad=True)
pi_r = torch.tensor([-3.0], requires_grad=True)  # Policy wrongly prefers rejected!
ref_c = torch.tensor([-4.0])
ref_r = torch.tensor([-4.0])

loss, _, _ = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1)
loss.backward()
print(f"dL/d(pi_chosen_logprob) = {pi_c.grad.item():.4f} (should be negative → increase chosen prob)")
print(f"dL/d(pi_rejected_logprob) = {pi_r.grad.item():.4f} (should be positive → decrease rejected prob)")

# Test 4: Equivalence to reward model loss with implicit rewards
print("\n" + "=" * 70)
print("TEST 4: DPO = Reward Model Loss on Implicit Rewards")
print("=" * 70)

pi_c = torch.tensor([-3.0, -2.0, -4.0])
pi_r = torch.tensor([-5.0, -6.0, -3.0])
ref_c = torch.tensor([-4.0, -3.0, -4.5])
ref_r = torch.tensor([-4.0, -5.0, -3.5])
beta = 0.1

loss_dpo, c_rew, r_rew = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=beta)

# Manually compute: it's just the preference loss on implicit rewards
implicit_c = beta * (pi_c - ref_c)
implicit_r = beta * (pi_r - ref_r)
loss_manual = -F.logsigmoid(implicit_c - implicit_r).mean()

print(f"DPO loss:    {loss_dpo.item():.6f}")
print(f"Manual:      {loss_manual.item():.6f}")
print(f"Match: {torch.isclose(loss_dpo, loss_manual)}")
```

**Common mistakes:**
1. Forgetting that log-probs are negative numbers (more negative = lower probability)
2. Getting the sign wrong in the log-ratio (should be policy - reference)
3. Not freezing the reference model during training
4. Confusing per-token vs sequence-level log-probs (DPO uses sequence-level)
5. Wrong beta direction (higher beta = more conservative, NOT more aggressive)
6. Using `log(sigmoid(x))` instead of `F.logsigmoid(x)` — numerically unstable

## Follow-up Questions
- Why does DPO not need a separate reward model?
- What happens when beta → 0? When beta → infinity?
- What is the "implicit reward" in DPO, and how does it relate to the RLHF reward?
- Why is the reference model needed? What goes wrong without it?
- What are the failure modes of DPO compared to RLHF with PPO?
- What is IPO (Identity Preference Optimization) and how does it differ?
