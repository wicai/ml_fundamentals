# Build and Train a Supervised Text Classifier
# ====================================================================
#
# Build a supervised text classifier from scratch by composing `nn.Module` sub-modules into a larger model, then train it end-to-end.
# 
# Your implementation should include:
# 1. **`EmbeddingBag`-based encoder**: Average token embeddings into a fixed-size sentence representation
# 2. **`Classifier` head**: MLP that maps sentence embeddings to class logits
# 3. **`TextClassifier`**: Composes the encoder + classifier into one `nn.Module`
# 4. **Training**: Train on synthetic data and verify the model learns
# 
# This tests your ability to:
# - Define custom `nn.Module` classes with `__init__` and `forward`
# - Compose smaller modules into a larger module
# - Use `nn.ModuleList`/sub-modules so parameters are properly registered
# - Wire up the full training pipeline (data → model → loss → optimizer → backward)
# 
# **Function signature:**
#
# ====================================================================
import torch
from torch import nn
from torch.nn.functional import F
class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: int = 0):
        """
        Encodes variable-length token sequences into fixed-size vectors
        using embedding averaging (like EmbeddingBag with mode='mean').

        Args:
            vocab_size: number of tokens in vocabulary
            embed_dim: embedding dimension
            padding_idx: index of padding token (excluded from average)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx)           

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, max_seq_len) padded token indices
        Returns:
            (batch, embed_dim) averaged embeddings
        """
        emb_per_token = self.embedding(token_ids) #(batch, max_seq_len, embed_dim)        
        # average across each sequence
        mask = (token_ids != self.padding_idx).float()  # 1 if we wanna count it, (batch, max_seq_len)
        tokens_per_seq = mask.sum(dim=-1).clamp(min=1) # (batch,) # clamp to avoid div by 0 if the whole seq is padding
        return torch.sum(emb_per_token, dim=1) / tokens_per_seq.unsqueeze(-1) # (batch, embed_dim) / (batch,)        

class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3):
        """
        MLP classification head: Linear → ReLU → Dropout → Linear.

        Args:
            input_dim: size of input features
            hidden_dim: hidden layer size
            num_classes: number of output classes
            dropout: dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_p = dropout
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(self.dropout_p)
        self.fc2 = nn.Linear(hidden_dim, num_classes)                

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            logits: (batch, num_classes)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_layer(x)
        return self.fc2(x)        

class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, padding_idx: int = 0, dropout: float = 0.3):
        """
        Full text classifier: encoder + classifier head.
        """
        super().__init__()        
        self.model = nn.Sequential(
            TextEncoder(vocab_size, embed_dim, padding_idx),
            ClassifierHead(embed_dim, hidden_dim, num_classes, dropout)
        )
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, max_seq_len)
        Returns:
            logits: (batch, num_classes)
        """
        return self.model(token_ids)

from torch import nn
import torch.nn.functional as F
def train_and_evaluate(num_epochs: int = 5) -> float:
      batch_size = 200
      seq_len = 20
      # 1. Create synthetic data
      # X is (batch_size, seq_len) # each value is the token_ind, from 1-100, where 0 is padding
      vocab_size = 100
      X = torch.randint(1, vocab_size, (batch_size, seq_len))
      num_classes = 10
      # Y is (batch_size,) # 0 to num_classes-1
      Y = torch.randint(0, num_classes, (batch_size,))
      # 2. Instantiate model
      model = TextClassifier(
        vocab_size=vocab_size,
        embed_dim = 1024,
        hidden_dim = 1024, 
        num_classes = num_classes,
        padding_idx = 0,
        dropout = 0.3
      )
      # 3. Define loss + optimizer
      loss_fn = nn.CrossEntropyLoss()
      optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)        
      # 4. Training loop
      for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, Y)
        loss.backward()
        optimizer.step()
      # 5. Return final accuracy
      # ideally we want hold-out set since this could be overfit but whatever for now
      with torch.no_grad():
        y_pred = model(X)  #(batch_size, num_classes)            
        y_pred_classes = torch.argmax(y_pred, dim=-1)
        return torch.mean((Y == y_pred_classes).float()).item()
