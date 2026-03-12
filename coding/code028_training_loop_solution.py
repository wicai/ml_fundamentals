# Implement a PyTorch Training Loop
# ====================================================================
#
# Implement a complete PyTorch training loop with proper train/eval mode handling.
# 
# Given a model, optimizer, loss function, and data loaders, write:
# 1. **`train_one_epoch`**: One full pass over the training data
# 2. **`evaluate`**: Evaluate on validation data (no gradients!)
# 3. **`training_loop`**: The outer loop that ties it all together
# 
# Your implementation must correctly:
# - Toggle `model.train()` and `model.eval()` at the right times
# - Use `torch.no_grad()` during evaluation
# - Call `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` in the right order
# - Track and return loss/accuracy metrics
# 
# **Function signature:**
#
# ====================================================================

def train_one_epoch(model, optimizer, loss_fn, train_loader):
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

def evaluate(model, loss_fn, val_loader):
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

def training_loop(model, optimizer, loss_fn, train_loader, val_loader, num_epochs):
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

