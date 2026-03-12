# Implement Reward Model Preference Loss

**Category:** coding
**Difficulty:** 3
**Tags:** coding, rlhf, reward model, alignment, safety

## Question
Implement the Bradley-Terry pairwise preference loss used to train reward models in RLHF.

Given a reward model that scores responses, train it so that preferred (chosen) responses get higher scores than rejected ones. This is the foundation of RLHF alignment.

Your implementation should include:
1. **`preference_loss`**: The core Bradley-Terry loss: `-log(sigmoid(r_chosen - r_rejected))`
2. **`reward_accuracy`**: What fraction of pairs does the model rank correctly?
3. **`train_reward_model_step`**: A single training step for a reward model

**Function signature:**
```python
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
    pass

def reward_accuracy(rewards_chosen: torch.Tensor, rewards_rejected: torch.Tensor) -> float:
    """
    Compute the fraction of pairs where the model correctly ranks chosen > rejected.

    Args:
        rewards_chosen: shape (batch_size,)
        rewards_rejected: shape (batch_size,)
    Returns:
        accuracy: float in [0, 1]
    """
    pass

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
    pass
```

## Answer

**Key concepts:**
1. We have human preference data: pairs of (chosen, rejected) responses
2. The Bradley-Terry model: P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
3. We maximize log-likelihood: minimize -log(sigmoid(r_chosen - r_rejected))
4. The loss pushes r_chosen up and r_rejected down simultaneously
5. Only the *difference* in rewards matters, not absolute values

**Reference implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def preference_loss(rewards_chosen: torch.Tensor, rewards_rejected: torch.Tensor) -> torch.Tensor:
    """
    Bradley-Terry pairwise preference loss.

    P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
    Loss = -log P(chosen > rejected) = -log sigmoid(r_chosen - r_rejected)

    Using F.logsigmoid for numerical stability (avoids log(sigmoid(x)) issues).
    """
    return -F.logsigmoid(rewards_chosen - rewards_rejected).mean()

def reward_accuracy(rewards_chosen: torch.Tensor, rewards_rejected: torch.Tensor) -> float:
    """Fraction of pairs correctly ranked."""
    return (rewards_chosen > rewards_rejected).float().mean().item()

def train_reward_model_step(model: nn.Module, optimizer: torch.optim.Optimizer, chosen_inputs: torch.Tensor, rejected_inputs: torch.Tensor) -> dict[str, float]:
    """Single training step for a reward model."""
    model.train()

    # Get scalar rewards for both responses
    r_chosen = model(chosen_inputs).squeeze(-1)    # (batch_size,)
    r_rejected = model(rejected_inputs).squeeze(-1)  # (batch_size,)

    # Compute preference loss
    loss = preference_loss(r_chosen, r_rejected)

    # Backprop and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = reward_accuracy(r_chosen.detach(), r_rejected.detach())

    return {'loss': loss.item(), 'accuracy': acc}
```

**Testing:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# Test 1: Loss properties
print("=" * 70)
print("TEST 1: Preference Loss Properties")
print("=" * 70)

# When chosen >> rejected, loss should be near 0
r_chosen = torch.tensor([5.0, 3.0, 10.0])
r_rejected = torch.tensor([1.0, 0.0, 2.0])
loss_easy = preference_loss(r_chosen, r_rejected)
print(f"Easy pairs (chosen >> rejected): loss = {loss_easy.item():.6f} (should be small)")

# When chosen == rejected, loss should be log(2) ≈ 0.693
r_equal = torch.tensor([1.0, 2.0, 3.0])
loss_equal = preference_loss(r_equal, r_equal)
print(f"Equal rewards: loss = {loss_equal.item():.6f} (should be ~{torch.log(torch.tensor(2.0)).item():.6f})")

# When chosen << rejected, loss should be large
loss_wrong = preference_loss(r_rejected, r_chosen)
print(f"Wrong ranking (chosen << rejected): loss = {loss_wrong.item():.6f} (should be large)")

print(f"Loss ordering correct: {loss_easy < loss_equal < loss_wrong}")

# Test 2: Accuracy
print("\n" + "=" * 70)
print("TEST 2: Reward Accuracy")
print("=" * 70)
r_chosen = torch.tensor([3.0, 2.0, 1.0, 4.0])
r_rejected = torch.tensor([1.0, 3.0, 0.5, 5.0])
acc = reward_accuracy(r_chosen, r_rejected)
print(f"Chosen:   {r_chosen.tolist()}")
print(f"Rejected: {r_rejected.tolist()}")
print(f"Accuracy: {acc:.2f} (should be 0.50 — 2 out of 4 correct)")

# Test 3: Training a reward model
print("\n" + "=" * 70)
print("TEST 3: Training a Reward Model")
print("=" * 70)

# Create a simple reward model
reward_model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)
optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.01)

# Synthetic preference data:
# chosen = positive features, rejected = negative features
torch.manual_seed(1)
chosen_data = torch.randn(64, 10) + 1.0   # shifted positive
rejected_data = torch.randn(64, 10) - 1.0  # shifted negative

# Train for a few steps
for step in range(50):
    metrics = train_reward_model_step(
        reward_model, optimizer, chosen_data, rejected_data
    )
    if step % 10 == 0:
        print(f"Step {step:3d}: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}")

print(f"\nFinal accuracy: {metrics['accuracy']:.4f} (should be near 1.0)")
print(f"Final loss: {metrics['loss']:.4f} (should be near 0.0)")

# Test 4: Gradient flows correctly
print("\n" + "=" * 70)
print("TEST 4: Gradient Direction")
print("=" * 70)
r_chosen = torch.tensor([1.0], requires_grad=True)
r_rejected = torch.tensor([2.0], requires_grad=True)  # Wrong ranking!

loss = preference_loss(r_chosen, r_rejected)
loss.backward()

print(f"r_chosen grad: {r_chosen.grad.item():.4f} (should be negative → push r_chosen UP)")
print(f"r_rejected grad: {r_rejected.grad.item():.4f} (should be positive → push r_rejected DOWN)")
print(f"Gradients push in correct direction: {r_chosen.grad.item() < 0 and r_rejected.grad.item() > 0}")

# Test 5: Compare with manual computation
print("\n" + "=" * 70)
print("TEST 5: Manual Verification")
print("=" * 70)
r_c = torch.tensor([2.0, 0.5])
r_r = torch.tensor([1.0, 1.5])
our_loss = preference_loss(r_c, r_r)
manual_loss = -torch.log(torch.sigmoid(r_c - r_r)).mean()
print(f"Our loss:    {our_loss.item():.6f}")
print(f"Manual:      {manual_loss.item():.6f}")
print(f"Match: {torch.isclose(our_loss, manual_loss)}")
```

**Common mistakes:**
1. Forgetting the negative sign (maximizing log-likelihood = minimizing negative log-likelihood)
2. Using `log(sigmoid(x))` instead of `F.logsigmoid(x)` — numerically unstable for large negative x
3. Not squeezing the reward model output from (batch, 1) to (batch,)
4. Confusing which is chosen vs rejected in the loss
5. Not understanding that only the reward *difference* matters — the model can add any constant

## Follow-up Questions
- Why does the Bradley-Terry model only care about reward differences?
- What is reward hacking, and how do reward models fail?
- How is the reward model used in PPO for RLHF?
- Why might you add a KL penalty term to prevent reward hacking?
- What's the connection between this loss and logistic regression?
