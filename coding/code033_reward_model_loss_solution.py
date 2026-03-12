# Implement Reward Model Preference Loss
# ====================================================================
#
# Implement the Bradley-Terry pairwise preference loss used to train reward models in RLHF.
# 
# Given a reward model that scores responses, train it so that preferred (chosen) responses get higher scores than rejected ones. This is the foundation of RLHF alignment.
# 
# Your implementation should include:
# 1. **`preference_loss`**: The core Bradley-Terry loss: `-log(sigmoid(r_chosen - r_rejected))`
# 2. **`reward_accuracy`**: What fraction of pairs does the model rank correctly?
# 3. **`train_reward_model_step`**: A single training step for a reward model
# 
# **Function signature:**
#
# ====================================================================

import torch
import torch.nn.functional as F
def preference_loss(rewards_chosen: torch.Tensor, rewards_rejected: torch.Tensor) -> torch.Tensor:
    """
    Compute the Bradley-Terry pairwise preference loss.

    Loss = -E[log(sigmoid(r_chosen - r_rejected))]

    This loss encourages the model to assign higher rewards to chosen responses.

    Args:
        rewards_chosen: scalar rewards for preferred responses, shape (batch_size,)
        rewards_rejected: scalar rewards for rejected responses, shape (batch_size,)
    Returns:
        loss: scalar loss value
    """
    diff = reward_chosen - rewards_rejected
    return -F.logsigmoid(diff).mean()

def reward_accuracy(rewards_chosen: torch.Tensor, rewards_rejected: torch.Tensor) -> float:
    """
    Compute the fraction of pairs where the model correctly ranks chosen > rejected.

    Args:
        rewards_chosen: shape (batch_size,)
        rewards_rejected: shape (batch_size,)
    Returns:
        accuracy: float in [0, 1]
    """
    return (rewards_chosen - rewards_rejected).float().mean().item()    

def train_reward_model_step(model: nn.Module, optimizer: torch.optim.Optimizer, chosen_inputs: torch.Tensor, rejected_inputs: torch.Tensor) -> dict[str, float]:
    """
    One training step for a reward model.

    The model takes input and returns a scalar reward. We train it so that
    model(chosen) > model(rejected) using the preference loss.

    Args:
        model: nn.Module that maps input → scalar reward
        optimizer: torch.optim.Optimizer
        chosen_inputs: batch of preferred inputs, shape (batch_size, input_dim)
        rejected_inputs: batch of rejected inputs, shape (batch_size, input_dim)
    Returns:
        dict with 'loss' and 'accuracy'
    """
    optimizer.zero_grad()
    # Compute the computational graph
    r_chosen = model(chosen_inputs) # (batch_size, )
    r_rejected = model(rejected_inputs)       
    loss = preference_loss(r_chosen, r_rejected) 
    # Populate the gradients by going backwards through the computational network
    loss.backward()
    # Actually change the parameters
    optimizer.step()
    accuracy = reward_accuracy(r_chosen, r_rejected)
    return {
        'loss': loss.item(),
        'accuracy': accuracy
    }    

