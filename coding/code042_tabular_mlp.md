# Build and Train a Tabular MLP Binary Classifier

**Category:** coding
**Difficulty:** 4
**Tags:** coding, nn.Module, nn.ModuleList, batch-norm, binary-classification, training, DataLoader

## Question

Build a binary classifier for tabular (numerical) data by stacking reusable `nn.Module` blocks using `nn.ModuleList`, then train it end-to-end with batched data.

Your implementation should include:
1. **`TabularBlock`**: A single residual-style block — `Linear → BatchNorm1d → ReLU → Dropout`
2. **`TabularMLP`**: Stacks N `TabularBlock`s via `nn.ModuleList`, followed by a final output `Linear`
3. **Training**: Use `DataLoader` for batching, call `model.train()` / `model.eval()` correctly, use `BCEWithLogitsLoss`

This tests your ability to:
- Use `nn.ModuleList` to register a dynamic list of sub-modules
- Understand why `model.train()` and `model.eval()` matter (BatchNorm and Dropout behave differently)
- Use `BCEWithLogitsLoss` for binary classification (vs `CrossEntropyLoss` for multi-class)
- Use `DataLoader` for batched training instead of feeding all data at once

**Function signature:**
```python
class TabularBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        """
        A single MLP block for tabular data.
        Architecture: Linear(dim → dim) → BatchNorm1d(dim) → ReLU → Dropout

        Args:
            dim: input and output dimension (same, so blocks are stackable)
            dropout: dropout rate
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, dim)
        Returns:
            (batch, dim)
        """
        pass


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_blocks: int, dropout: float = 0.3):
        """
        MLP for tabular binary classification.
        Architecture:
            Linear(input_dim → hidden_dim)
            → N x TabularBlock(hidden_dim)     # stored in nn.ModuleList
            → Linear(hidden_dim → 1)           # output logit

        Args:
            input_dim: number of input features
            hidden_dim: hidden dimension (used throughout)
            num_blocks: how many TabularBlocks to stack
            dropout: dropout rate passed to each block
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            logits: (batch, 1) — raw logit (no sigmoid)
        """
        pass


def train_and_evaluate(num_epochs: int = 10) -> float:
    """
    Train TabularMLP on synthetic binary classification data.
    Returns validation accuracy as a float in [0, 1].

    Requirements:
    - Generate synthetic data: 1000 train, 200 val samples, 20 features each
    - Use DataLoader with batch_size=32 for training
    - Call model.train() before training loop, model.eval() before evaluation
    - Use BCEWithLogitsLoss (not CrossEntropyLoss)
    - Use AdamW optimizer
    - Return validation accuracy
    """
    pass
```

## Answer

**Key concepts:**
1. `nn.ModuleList` registers a Python list of modules so their parameters show up in `model.parameters()`. A plain `list` would NOT register them.
2. `model.train()` sets BatchNorm to use batch statistics; `model.eval()` uses running statistics accumulated during training. Dropout is also disabled in eval mode.
3. `BCEWithLogitsLoss` expects raw logits of shape `(batch, 1)` and float targets `(batch, 1)`. It applies sigmoid internally — never apply sigmoid before passing to it.
4. `DataLoader` shuffles and batches data automatically. More realistic than feeding all data at once.

**Reference implementation:**
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TabularBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_blocks: int, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            TabularBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x)  # (batch, 1)


def train_and_evaluate(num_epochs: int = 10) -> float:
    torch.manual_seed(1)

    # Synthetic binary classification: label = 1 if sum of first 5 features > 0
    def make_data(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        X = torch.randn(n, 20)
        y = (X[:, :5].sum(dim=1) > 0).float().unsqueeze(-1)  # (n, 1)
        return X, y

    X_train, y_train = make_data(1000)
    X_val, y_val = make_data(200)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    model = TabularMLP(input_dim=20, hidden_dim=64, num_blocks=3, dropout=0.3)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_val)
        preds = (logits > 0).float()  # threshold at 0 (equivalent to sigmoid > 0.5)
        accuracy = (preds == y_val).float().mean().item()

    return accuracy
```

**Common mistakes:**
1. Storing blocks in a plain `list` instead of `nn.ModuleList` — parameters won't be registered
2. Forgetting `model.train()` before training and `model.eval()` before evaluation — BatchNorm will use wrong statistics
3. Using `CrossEntropyLoss` for binary classification — works but expects 2-class logits `(batch, 2)` not `(batch, 1)`
4. Applying `torch.sigmoid` before `BCEWithLogitsLoss` — double-squashing kills gradients
5. Thresholding at 0.5 on logits instead of on sigmoid output — threshold on logits should be 0

## Follow-up Questions
- What's the difference between `nn.ModuleList` and a plain Python `list` of modules?
- Why does `model.eval()` matter even when you're using `torch.no_grad()`?
- Why is `BCEWithLogitsLoss` preferred over `BCELoss` (which takes probabilities)?
- What does BatchNorm actually normalize, and why does it behave differently at train vs eval time?
- How would you add early stopping based on validation loss?
