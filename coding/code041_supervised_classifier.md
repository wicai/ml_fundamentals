# Build and Train a Supervised Text Classifier

**Category:** coding
**Difficulty:** 4
**Tags:** coding, nn.Module, classification, training, end-to-end

## Question
Build a supervised text classifier from scratch by composing `nn.Module` sub-modules into a larger model, then train it end-to-end.

Your implementation should include:
1. **`EmbeddingBag`-based encoder**: Average token embeddings into a fixed-size sentence representation
2. **`Classifier` head**: MLP that maps sentence embeddings to class logits
3. **`TextClassifier`**: Composes the encoder + classifier into one `nn.Module`
4. **Training**: Train on synthetic data and verify the model learns

This tests your ability to:
- Define custom `nn.Module` classes with `__init__` and `forward`
- Compose smaller modules into a larger module
- Use `nn.ModuleList`/sub-modules so parameters are properly registered
- Wire up the full training pipeline (data → model → loss → optimizer → backward)

**Function signature:**
```python
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
        pass

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, max_seq_len) padded token indices
        Returns:
            (batch, embed_dim) averaged embeddings
        """
        pass

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
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            logits: (batch, num_classes)
        """
        pass

class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, padding_idx: int = 0, dropout: float = 0.3):
        """
        Full text classifier: encoder + classifier head.
        """
        pass

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, max_seq_len)
        Returns:
            logits: (batch, num_classes)
        """
        pass
```

## Answer

**Key concepts:**
1. Each sub-component is its own `nn.Module` with `__init__` + `forward`
2. Assigning sub-modules as attributes in `__init__` (via `self.x = nn.Linear(...)`) auto-registers their parameters
3. The parent module composes children — calling `model.parameters()` recursively collects all params
4. Padding tokens must be excluded from the embedding average
5. Return raw logits (not softmax) — `nn.CrossEntropyLoss` expects logits

**Reference implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.padding_idx = padding_idx

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq_len)
        embedded = self.embedding(token_ids)  # (batch, seq_len, embed_dim)

        # Mask out padding tokens before averaging
        mask = (token_ids != self.padding_idx).unsqueeze(-1).float()  # (batch, seq_len, 1)
        masked_embedded = embedded * mask

        # Average over non-padding tokens
        lengths = mask.sum(dim=1).clamp(min=1)  # (batch, 1) — clamp to avoid div by zero
        pooled = masked_embedded.sum(dim=1) / lengths  # (batch, embed_dim)

        return pooled

class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # Raw logits — no softmax!

class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, padding_idx: int = 0, dropout: float = 0.3):
        super().__init__()
        self.encoder = TextEncoder(vocab_size, embed_dim, padding_idx)
        self.head = ClassifierHead(embed_dim, hidden_dim, num_classes, dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        features = self.encoder(token_ids)   # (batch, embed_dim)
        logits = self.head(features)          # (batch, num_classes)
        return logits
```

**Testing:**
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(1)

# --- Synthetic dataset ---
# 3 classes, each defined by which token indices appear most
VOCAB_SIZE = 50
NUM_CLASSES = 3
SEQ_LEN = 12
N_TRAIN = 600
N_VAL = 150

def make_data(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Each class uses tokens from a distinct range, plus noise."""
    tokens_list = []
    labels = []
    for i in range(n):
        label = i % NUM_CLASSES
        # Class 0 uses tokens 1-15, class 1 uses 16-30, class 2 uses 31-45
        low = label * 15 + 1
        high = low + 15
        seq = torch.randint(low, high, (SEQ_LEN - 2,))
        # Add some noise tokens and padding
        noise = torch.randint(1, VOCAB_SIZE, (2,))
        seq = torch.cat([seq, noise])
        tokens_list.append(seq)
        labels.append(label)
    return torch.stack(tokens_list), torch.tensor(labels)

X_train, y_train = make_data(N_TRAIN)
X_val, y_val = make_data(N_VAL)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

# --- Test 1: Module composition ---
print("=" * 70)
print("TEST 1: Module Composition")
print("=" * 70)
model = TextClassifier(VOCAB_SIZE, embed_dim=64, hidden_dim=32, num_classes=NUM_CLASSES)

# Verify sub-modules are registered
named_children = dict(model.named_children())
print(f"Sub-modules: {list(named_children.keys())}")
assert 'encoder' in named_children, "encoder should be a registered sub-module"
assert 'head' in named_children, "head should be a registered sub-module"

# Verify parameters are collected recursively
num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {num_params:,}")
assert num_params > 0, "Model should have trainable parameters"

# --- Test 2: Forward pass shapes ---
print("\n" + "=" * 70)
print("TEST 2: Forward Pass")
print("=" * 70)
batch = X_train[:4]
logits = model(batch)
print(f"Input shape:  {batch.shape}")
print(f"Output shape: {logits.shape}")
assert logits.shape == (4, NUM_CLASSES), f"Expected (4, {NUM_CLASSES}), got {logits.shape}"

# --- Test 3: Training loop — loss decreases ---
print("\n" + "=" * 70)
print("TEST 3: Training — Loss Should Decrease")
print("=" * 70)
model = TextClassifier(VOCAB_SIZE, embed_dim=64, hidden_dim=32, num_classes=NUM_CLASSES)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

losses = []
for epoch in range(15):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += inputs.size(0)

    avg_loss = epoch_loss / total
    accuracy = correct / total
    losses.append(avg_loss)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d} — loss: {avg_loss:.4f}, acc: {accuracy:.4f}")

print(f"\nFirst loss: {losses[0]:.4f}")
print(f"Final loss: {losses[-1]:.4f}")
print(f"Loss decreased: {losses[-1] < losses[0]}")
assert losses[-1] < losses[0], "Training loss should decrease!"

# --- Test 4: Evaluation ---
print("\n" + "=" * 70)
print("TEST 4: Validation Accuracy")
print("=" * 70)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in val_loader:
        logits = model(inputs)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += inputs.size(0)

val_acc = correct / total
print(f"Validation accuracy: {val_acc:.4f}")
print(f"Better than random ({1/NUM_CLASSES:.4f}): {val_acc > 1/NUM_CLASSES}")
assert val_acc > 1 / NUM_CLASSES + 0.1, "Should do significantly better than random"

# --- Test 5: Verify padding handling ---
print("\n" + "=" * 70)
print("TEST 5: Padding Handling")
print("=" * 70)
padded = torch.tensor([[5, 10, 0, 0, 0],
                        [5, 10, 15, 0, 0]])
encoder = model.encoder
out1 = encoder(padded[:1])
out2 = encoder(padded[1:])
print(f"Seq [5,10,pad,pad,pad] embedding norm: {out1.norm():.4f}")
print(f"Seq [5,10,15,pad,pad] embedding norm: {out2.norm():.4f}")
print("Padding tokens should NOT affect the embedding average")
# Verify: a sequence of [5, 10] should give different result than [5, 10, 15]
assert not torch.allclose(out1, out2), "Different non-padding tokens should give different embeddings"
```

**Common mistakes:**
1. Forgetting `super().__init__()` — parameters won't be registered
2. Storing sub-modules in a plain Python list instead of `nn.ModuleList` — parameters invisible to `model.parameters()`
3. Applying softmax before returning logits — `nn.CrossEntropyLoss` already does softmax internally
4. Not masking padding tokens in the encoder — padding pollutes the average
5. Dividing by `seq_len` instead of number of non-padding tokens

## Follow-up Questions
- What happens if you store sub-modules in a plain `list` instead of as `self.x = ...` attributes?
- Why return logits instead of probabilities?
- How would you add a second encoder (e.g., for title + body) and merge their outputs?
- What's the difference between `nn.EmbeddingBag` and manually averaging `nn.Embedding`?
