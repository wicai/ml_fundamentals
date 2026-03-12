import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_policy_ratio(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
) -> torch.Tensor:
    return torch.exp(log_probs_new - log_probs_old)


def ppo_clip_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    ratio = compute_policy_ratio(log_probs_new, log_probs_old)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()

    info = {
        'clip_fraction': ((ratio - 1).abs() > clip_eps).float().mean().item(),
        'approx_kl': (log_probs_old - log_probs_new).mean().item(),
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
    batch, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0

    for t in reversed(range(seq_len)):
        not_done = 1.0 - dones[:, t]
        delta = rewards[:, t] + gamma * values[:, t + 1] * not_done - values[:, t]
        last_gae = delta + gamma * lam * not_done * last_gae
        advantages[:, t] = last_gae

    returns = advantages + values[:, :-1]
    return advantages, returns


def _get_log_probs(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids)
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lps = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_lps.sum(dim=-1)


def _get_entropy(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids)[:, :-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
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
    metrics = {k: 0.0 for k in ['policy_loss', 'value_loss', 'entropy', 'clip_fraction', 'approx_kl']}

    for _ in range(n_epochs):
        policy_model.train()
        value_model.train()

        log_probs_new = _get_log_probs(policy_model, input_ids)
        policy_loss, info = ppo_clip_loss(log_probs_new, log_probs_old.detach(), advantages, clip_eps)

        values_pred = value_model(input_ids)
        value_loss = F.mse_loss(values_pred, returns)

        entropy = _get_entropy(policy_model, input_ids)
        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics['policy_loss'] += policy_loss.item()
        metrics['value_loss'] += value_loss.item()
        metrics['entropy'] += entropy.item()
        metrics['clip_fraction'] += info['clip_fraction']
        metrics['approx_kl'] += info['approx_kl']

    return {k: v / n_epochs for k, v in metrics.items()}
