# Build and Train a Tabular MLP Binary Classifier
# ====================================================================
#
# Build a binary classifier for tabular (numerical) data by stacking reusable `nn.Module` blocks using `nn.ModuleList`, then train it end-to-end with batched data.
# 
# Your implementation should include:
# 1. **`TabularBlock`**: A single residual-style block — `Linear → BatchNorm1d → ReLU → Dropout`
# 2. **`TabularMLP`**: Stacks N `TabularBlock`s via `nn.ModuleList`, followed by a final output `Linear`
# 3. **Training**: Use `DataLoader` for batching, call `model.train()` / `model.eval()` correctly, use `BCEWithLogitsLoss`
# 
# This tests your ability to:
# - Use `nn.ModuleList` to register a dynamic list of sub-modules
# - Understand why `model.train()` and `model.eval()` matter (BatchNorm and Dropout behave differently)
# - Use `BCEWithLogitsLoss` for binary classification (vs `CrossEntropyLoss` for multi-class)
# - Use `DataLoader` for batched training instead of feeding all data at once
# 
# **Function signature:**
#
# ====================================================================

import torch
from torch import nn
from torch.nn import functional as F 
class TabularBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        """
        A single MLP block for tabular data.
        Architecture: Linear(dim → dim) → BatchNorm1d(dim) → ReLU → Dropout

        Args:
            dim: input and output dimension (same, so blocks are stackable)
            dropout: dropout rate
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, dim)
        Returns:
            (batch, dim)
        """
        return self.block(x)        


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
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([TabularBlock(hidden_dim, dropout) for i in range(num_blocks)])
        self.fc2 = nn.Linear(hidden_dim, 1)        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            logits: (batch, 1) — raw logit (no sigmoid)
        """
        x = self.fc1(x)
        for block in self.blocks:
            x = block(x)
        return self.fc2(x)        

from torch.utils.data import DataLoader, TensorDataset
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
    torch.manual_seed(1)
    # Generate synthetic data: 1000 train, 200 val samples, 20 features each
    # (n_features)
    n_train = 1000
    n_val = 200
    n_features = 20
    X_train = torch.randn((n_train, n_features))
    X_val = torch.randn((n_val, n_features))
    Y_train = torch.randint(0, 2, (n_train, 1)).float()
    Y_val = torch.randint(0, 2, (n_val, 1)).float()
    train_ds = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_ds, batch_size = 32, shuffle=True)
    model = TabularMLP(
        input_dim=n_features, 
        hidden_dim=64,
        num_blocks=3,
        dropout=.3
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.02)
    
    for epoch in range(num_epochs):
        model.train()
        # loop through batches
        for X_batch, Y_batch in train_loader:
            # zero grad
            optimizer.zero_grad()
            # do forward
            y_pred_batch = model(X_batch)            
            # compute loss
            loss = criterion(y_pred_batch, Y_batch)
            # call backwards on the loss tensor
            loss.backward()
            # take a step for the optimizer
            optimizer.step()
        with torch.no_grad():
            model.eval()
            y_pred_eval = model(X_val) # logits        
            y_pred_binary = (y_pred_eval > 0).float()
            epoch_accuracy = torch.mean((y_pred_binary == Y_val).float())
            print(f'After epoch {epoch} accuracy is {epoch_accuracy}')
    return epoch_accuracy