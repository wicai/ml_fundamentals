# Implement Gradient Accumulation

**Category:** coding
**Difficulty:** 3
**Tags:** coding, training, optimization, gradients

## Question
Implement gradient accumulation to simulate a larger effective batch size when GPU memory is limited.

Instead of updating weights every mini-batch, accumulate gradients over N mini-batches before stepping. This gives the same result as using a batch size of `N * mini_batch_size`.

Your implementation should:
1. Accumulate gradients over `accumulation_steps` mini-batches
2. Scale the loss correctly so the effective loss matches a single large batch
3. Only call `optimizer.step()` and `optimizer.zero_grad()` every N steps

**Function signature:**
```python
def train_one_epoch_with_accumulation(model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, train_loader: DataLoader, accumulation_steps: int) -> dict[str, float]:
    """
    Train for one epoch with gradient accumulation.

    Args:
        model: nn.Module
        optimizer: torch.optim.Optimizer
        loss_fn: loss function (with default 'mean' reduction)
        train_loader: DataLoader yielding (inputs, targets)
        accumulation_steps: number of mini-batches to accumulate before stepping
    Returns:
        dict with 'loss' (avg training loss)
    """
    pass
```

## Answer

**Key concepts:**
1. Gradients accumulate by default in PyTorch (`.backward()` adds to `.grad`)
2. Divide loss by `accumulation_steps` to get correct average
3. Only call `optimizer.step()` + `optimizer.zero_grad()` every N steps
4. Handle the last incomplete accumulation group at the end of the epoch

**Why it works:**
- Normal: `loss = mean(batch)`, then `backward()` once
- Accumulated: `loss = mean(mini_batch) / N`, then `backward()` N times
- The gradients sum up, giving the same result as processing all N mini-batches at once

**Reference implementation:**
```python
import torch
import torch.nn as nn

def train_one_epoch_with_accumulation(model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, train_loader: DataLoader, accumulation_steps: int) -> dict[str, float]:
    """Train with gradient accumulation."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    optimizer.zero_grad()  # Clear gradients at the start

    for step, (inputs, targets) in enumerate(train_loader):
        device = next(model.parameters()).device
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Scale loss by accumulation steps
        # This ensures the gradient magnitude matches a full-sized batch
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()  # Gradients ACCUMULATE (added to .grad)

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

        # Step only every accumulation_steps, or at the last batch
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

    return {'loss': total_loss / total_samples}
```

**Testing:**
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(1)

# Create dataset
X = torch.randn(256, 10)
y = (X[:, 0] > 0).long()

# Test 1: Gradient accumulation matches large batch training
print("=" * 70)
print("TEST 1: Accumulation Matches Large Batch")
print("=" * 70)

# Approach A: Large batch (batch_size=64)
model_a = nn.Linear(10, 2)
torch.manual_seed(1)
nn.init.xavier_uniform_(model_a.weight)
nn.init.zeros_(model_a.bias)
opt_a = torch.optim.SGD(model_a.parameters(), lr=0.1)
loader_a = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=False)

loss_fn = nn.CrossEntropyLoss()
model_a.train()
for inputs, targets in loader_a:
    opt_a.zero_grad()
    loss = loss_fn(model_a(inputs), targets)
    loss.backward()
    opt_a.step()
    break  # Just one large batch

# Approach B: Small batch (batch_size=16) with accumulation_steps=4
model_b = nn.Linear(10, 2)
torch.manual_seed(1)
nn.init.xavier_uniform_(model_b.weight)
nn.init.zeros_(model_b.bias)
opt_b = torch.optim.SGD(model_b.parameters(), lr=0.1)

# Manually simulate: process 4 mini-batches of 16 = effective batch of 64
loader_b = DataLoader(TensorDataset(X[:64], y[:64]), batch_size=16, shuffle=False)

model_b.train()
opt_b.zero_grad()
for i, (inputs, targets) in enumerate(loader_b):
    loss = loss_fn(model_b(inputs), targets) / 4  # Scale by accum steps
    loss.backward()

opt_b.step()

# Compare parameters
weight_diff = (model_a.weight - model_b.weight).abs().max().item()
bias_diff = (model_a.bias - model_b.bias).abs().max().item()
print(f"Weight difference: {weight_diff:.2e}")
print(f"Bias difference:   {bias_diff:.2e}")
print(f"Match: {weight_diff < 1e-5 and bias_diff < 1e-5}")

# Test 2: Full function test
print("\n" + "=" * 70)
print("TEST 2: train_one_epoch_with_accumulation")
print("=" * 70)
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)

metrics = train_one_epoch_with_accumulation(model, optimizer, loss_fn, loader, accumulation_steps=4)
print(f"Loss: {metrics['loss']:.4f}")
print(f"Effective batch size: {16 * 4} = 64")

# Test 3: Verify gradients are zeroed after step
print("\n" + "=" * 70)
print("TEST 3: Gradients Zeroed After Step")
print("=" * 70)
model = nn.Linear(10, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=False)

train_one_epoch_with_accumulation(model, optimizer, loss_fn, loader, accumulation_steps=2)
grad_is_zero = all(
    p.grad is None or p.grad.abs().max() == 0
    for p in model.parameters()
)
print(f"Gradients zeroed after epoch: {grad_is_zero}")

# Test 4: Accumulation steps = 1 should behave like normal training
print("\n" + "=" * 70)
print("TEST 4: accumulation_steps=1 Matches Normal Training")
print("=" * 70)
print("accumulation_steps=1 is just standard training (step every batch)")
```

**Common mistakes:**
1. Forgetting to divide loss by `accumulation_steps` (gradients too large)
2. Calling `zero_grad()` every step instead of every N steps (defeats the purpose)
3. Forgetting to handle the last incomplete group of batches
4. Not calling `zero_grad()` at the start of the epoch
5. Using `loss.item()` on the scaled loss instead of the original for logging

## Follow-up Questions
- Why not just use a larger batch size directly? (GPU memory!)
- How does gradient accumulation interact with batch normalization?
- Does gradient accumulation give exactly the same result as large batches? (Yes for SGD, approximately for Adam due to denominator)
- How does this relate to distributed training / data parallelism?
