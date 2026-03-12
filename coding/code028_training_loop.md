# Implement a PyTorch Training Loop

**Category:** coding
**Difficulty:** 3
**Tags:** coding, training, pytorch, backprop

## Question
Implement a complete PyTorch training loop with proper train/eval mode handling.

Given a model, optimizer, loss function, and data loaders, write:
1. **`train_one_epoch`**: One full pass over the training data
2. **`evaluate`**: Evaluate on validation data (no gradients!)
3. **`training_loop`**: The outer loop that ties it all together

Your implementation must correctly:
- Toggle `model.train()` and `model.eval()` at the right times
- Use `torch.no_grad()` during evaluation
- Call `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` in the right order
- Track and return loss/accuracy metrics

**Function signature:**
```python
def train_one_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, train_loader: DataLoader) -> dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        model: nn.Module
        optimizer: torch.optim.Optimizer
        loss_fn: loss function (e.g., nn.CrossEntropyLoss())
        train_loader: DataLoader yielding (inputs, targets)
    Returns:
        dict with 'loss' (avg training loss) and 'accuracy' (training accuracy)
    """
    pass

def evaluate(model: nn.Module, loss_fn: nn.Module, val_loader: DataLoader) -> dict[str, float]:
    """
    Evaluate the model on validation data. No gradient computation!

    Args:
        model: nn.Module
        loss_fn: loss function
        val_loader: DataLoader yielding (inputs, targets)
    Returns:
        dict with 'loss' (avg val loss) and 'accuracy' (val accuracy)
    """
    pass

def training_loop(model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int) -> list[dict[str, float]]:
    """
    Full training loop over multiple epochs.

    Args:
        model, optimizer, loss_fn: as above
        train_loader, val_loader: DataLoaders
        num_epochs: number of epochs to train
    Returns:
        history: list of dicts, one per epoch, with train_loss, train_acc,
                 val_loss, val_acc
    """
    pass
```

## Answer

**Key concepts:**
1. `model.train()` enables dropout and uses batch stats for batchnorm
2. `model.eval()` disables dropout and uses running stats for batchnorm
3. `torch.no_grad()` disables gradient tracking (saves memory during eval)
4. Order matters: zero_grad → forward → loss → backward → step
5. Always move data to the same device as the model

**Reference implementation:**
```python
import torch
import torch.nn as nn

def train_one_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, train_loader: DataLoader) -> dict[str, float]:
    """Train for one epoch, returning metrics."""
    model.train()  # Enable dropout, use batch stats for batchnorm

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        # Move to same device as model
        device = next(model.parameters()).device
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Backward pass
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute new gradients
        optimizer.step()       # Update parameters

        # Track metrics
        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += inputs.size(0)

    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
    }

def evaluate(model: nn.Module, loss_fn: nn.Module, val_loader: DataLoader) -> dict[str, float]:
    """Evaluate model — no gradients, eval mode."""
    model.eval()  # Disable dropout, use running stats for batchnorm

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Don't track gradients — saves memory
        for inputs, targets in val_loader:
            device = next(model.parameters()).device
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)

    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
    }

def training_loop(model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int) -> list[dict[str, float]]:
    """Full training loop with train + eval each epoch."""
    history = []

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(model, optimizer, loss_fn, train_loader)
        val_metrics = evaluate(model, loss_fn, val_loader)

        epoch_result = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
        }
        history.append(epoch_result)

        print(f"Epoch {epoch+1}/{num_epochs} — "
              f"train_loss: {train_metrics['loss']:.4f}, "
              f"train_acc: {train_metrics['accuracy']:.4f}, "
              f"val_loss: {val_metrics['loss']:.4f}, "
              f"val_acc: {val_metrics['accuracy']:.4f}")

    return history
```

**Testing:**
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Create a simple dataset (XOR-like, 4 classes)
torch.manual_seed(1)
X_train = torch.randn(500, 10)
y_train = (X_train[:, 0] > 0).long() * 2 + (X_train[:, 1] > 0).long()
X_val = torch.randn(100, 10)
y_val = (X_val[:, 0] > 0).long() * 2 + (X_val[:, 1] > 0).long()

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

# Model with dropout + batchnorm (to verify train/eval mode matters)
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(32, 4),
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Test 1: train_one_epoch returns reasonable metrics
print("=" * 70)
print("TEST 1: Training One Epoch")
print("=" * 70)
metrics = train_one_epoch(model, optimizer, loss_fn, train_loader)
print(f"Train loss: {metrics['loss']:.4f}")
print(f"Train acc:  {metrics['accuracy']:.4f}")
assert 'loss' in metrics and 'accuracy' in metrics, "Must return loss and accuracy"
assert metrics['loss'] > 0, "Loss should be positive"
assert 0 <= metrics['accuracy'] <= 1, "Accuracy should be in [0, 1]"

# Test 2: evaluate returns metrics and doesn't update model
print("\n" + "=" * 70)
print("TEST 2: Evaluation")
print("=" * 70)
params_before = {n: p.clone() for n, p in model.named_parameters()}
val_metrics = evaluate(model, loss_fn, val_loader)
params_after = {n: p for n, p in model.named_parameters()}
print(f"Val loss: {val_metrics['loss']:.4f}")
print(f"Val acc:  {val_metrics['accuracy']:.4f}")
params_unchanged = all(
    torch.equal(params_before[n], params_after[n])
    for n in params_before
)
print(f"Parameters unchanged during eval: {params_unchanged}")

# Test 3: Full training loop shows improvement
print("\n" + "=" * 70)
print("TEST 3: Full Training Loop")
print("=" * 70)
model2 = nn.Sequential(
    nn.Linear(10, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(32, 4),
)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)
history = training_loop(model2, optimizer2, loss_fn, train_loader, val_loader, num_epochs=10)
print(f"\nFirst epoch loss: {history[0]['train_loss']:.4f}")
print(f"Last epoch loss:  {history[-1]['train_loss']:.4f}")
print(f"Loss decreased: {history[-1]['train_loss'] < history[0]['train_loss']}")
print(f"Final val accuracy: {history[-1]['val_acc']:.4f}")

# Test 4: Verify train/eval mode is set correctly
print("\n" + "=" * 70)
print("TEST 4: Train/Eval Mode Verification")
print("=" * 70)
train_one_epoch(model2, optimizer2, loss_fn, train_loader)
print(f"After train_one_epoch, model.training = {model2.training} (should be True)")
evaluate(model2, loss_fn, val_loader)
print(f"After evaluate, model.training = {model2.training} (should be False)")
```

**Common mistakes:**
1. Forgetting `model.train()` before training (dropout/batchnorm wrong)
2. Forgetting `model.eval()` before evaluation (inconsistent results)
3. Forgetting `torch.no_grad()` during eval (wastes memory)
4. Calling `optimizer.step()` before `loss.backward()` (no gradients yet!)
5. Calling `loss.backward()` before `optimizer.zero_grad()` (accumulates old gradients)
6. Using `loss` instead of `loss.item()` for tracking (keeps graph in memory)
7. Not moving data to the same device as the model

## Follow-up Questions
- Why does the order zero_grad → forward → backward → step matter?
- What happens if you forget `model.eval()` during validation?
- What's the difference between `optimizer.zero_grad()` and `model.zero_grad()`?
- Why use `loss.item()` instead of just `loss` when accumulating?
- When would you call `zero_grad(set_to_none=True)` and why?
