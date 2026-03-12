# Implement Gradient Accumulation
# ====================================================================
#
# Implement gradient accumulation to simulate a larger effective batch size when GPU memory is limited.
# 
# Instead of updating weights every mini-batch, accumulate gradients over N mini-batches before stepping. This gives the same result as using a batch size of `N * mini_batch_size`.
# 
# Your implementation should:
# 1. Accumulate gradients over `accumulation_steps` mini-batches
# 2. Scale the loss correctly so the effective loss matches a single large batch
# 3. Only call `optimizer.step()` and `optimizer.zero_grad()` every N steps
# 
# **Function signature:**
#
# ====================================================================

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

