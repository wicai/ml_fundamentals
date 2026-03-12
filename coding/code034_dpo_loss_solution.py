# Implement DPO (Direct Preference Optimization) Loss
# ====================================================================
#
# Implement the DPO loss function. DPO skips training a separate reward model and directly optimizes the policy using preference data.
# 
# The key insight: the optimal policy under a KL-constrained reward maximization objective can be expressed as a closed-form function of the policy's own log-probabilities relative to a reference model.
# 
# Your implementation should include:
# 1. **`dpo_loss`**: The core DPO objective
# 2. **`compute_log_ratio`**: Helper to compute log(pi/ref) for a sequence
# 3. **`dpo_training_step`**: A single training step
# 
# **Function signature:**
#
# ====================================================================
import torch
from torch import nn
import torch.nn.functional as F

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
    return (
        policy_chosen_logprobs - ref_chosen_logprobs,
        policy_rejected_logprobs - ref_rejected_logprobs,
    )

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
    chosen_lr, rejected_lr = compute_log_ratio(policy_chosen_logprobs, policy_rejected_logprobs, ref_chosen_logprobs, ref_rejected_logprobs)    
    chosen_rewards = chosen_lr * beta
    rejected_rewards = rejected_lr * beta     
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    # chosen_rewards is high if pi chooses chosen response more than ref
    # rejected_reward is low (making the inside of logsigmoid high) if pi chooses chosen response more than ref
    # sigmoid maps log odds to [0,1] monotonically, and log is also monotonic but maps to [-infty, 0]
    # so F... will be high if pi is much better than ref at generating the chosen and avoiding the rejected
    # so, we need to negate it for loss 
    return (loss, chosen_rewards, rejected_rewards)

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
    # all of these are  (batch_size, seq_len, vocab_size)
    optimizer.zero_grad()
    with torch.no_grad():
        ref_chosen_logits = ref_model(chosen_ids) 
        ref_rejected_logits = ref_model(rejected_ids)
    policy_chosen_logits = policy_model(chosen_ids)
    policy_rejected_logits = policy_model(rejected_ids)
    # need to turn them into log_prob for each target id in chosen_ids[i+1]
    lp = F.log_softmax(ref_chosen_logits[:,:-1,:], axis=-1) # (batch_size, seq_len-1, vocab_len) each entry is a logprob
    ref_chosen_logprobs = lp.gather(2, chosen_ids[:,1:].unsqueeze(-1)).squeeze(-1) # (batch, seq_len-1), each entry is a logprob    
    lp = F.log_softmax(ref_rejected_logits[:,:-1,:], axis=-1) # (batch_size, seq_len, vocab_len) each entry is a logprob
    ref_rejected_logprobs = lp.gather(2, rejected_ids[:,1:].unsqueeze(-1)).squeeze(-1)
    lp = F.log_softmax(policy_chosen_logits[:,:-1,:], axis=-1) # (batch_size, seq_len, vocab_len) each entry is a logprob
    policy_chosen_logprobs = lp.gather(2, chosen_ids[:,1:].unsqueeze(-1)).squeeze(-1)
    lp = F.log_softmax(policy_rejected_logits[:,:-1,:], axis=-1) # (batch_size, seq_len, vocab_len) each entry is a logprob    
    policy_rejected_logprobs = lp.gather(2, rejected_ids[:, 1:].unsqueeze(-1)).squeeze(-1) # (batch_size, seq_len - 1)
    results = {}
    loss, chosen_reward, rejected_reward = dpo_loss(
        torch.sum(policy_chosen_logprobs, axis=-1),
        torch.sum(policy_rejected_logprobs, axis=-1),
        torch.sum(ref_chosen_logprobs, axis=-1),
        torch.sum(ref_rejected_logprobs, axis=-1),
        beta
    )
    results['loss'] = loss.item()
    results['chosen_reward'] = chosen_reward.mean().item()
    results['rejected_reward'] = rejected_reward.mean().item()
    results['accuracy'] = (chosen_reward - rejected_reward > 0).float().mean().item()
    loss.backward()
    optimizer.step()
    return results

