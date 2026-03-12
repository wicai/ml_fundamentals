# Implement Linear Probing for Interpretability

**Category:** coding
**Difficulty:** 3
**Tags:** coding, interpretability, linear-probe, mechanistic-interpretability, safety

## Question

Implement linear probing — a core interpretability technique that tests whether a concept is linearly encoded in a model's hidden states. If a simple linear classifier can predict a concept from a layer's activations, that layer "represents" the concept.

**Why this matters for safety:**
- Probing can reveal whether a model has learned to represent dangerous concepts internally, even if it doesn't express them in output
- It's used to study whether models represent things like "I'm being monitored," "this is a harmful request," or honesty/deception
- Understanding which layers encode which concepts informs targeted interventions (steering, fine-tuning, circuit analysis)

Your implementation should include:
1. **`extract_hidden_states`**: Pull activations from a specific layer for a batch of inputs
2. **`train_probe`**: Fit a logistic regression on hidden states to predict a binary label
3. **`probe_accuracy`**: Evaluate the probe on held-out data
4. **`layer_probe_sweep`**: Find which layer best encodes a concept by training probes at every layer

**Function signature:**
```python
def extract_hidden_states(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int,
    position: int | str = 'last',
) -> torch.Tensor:
    """
    Extract hidden states from a specific transformer layer.

    Args:
        model: transformer model with model.layers[i] interface
        input_ids: shape (batch, seq_len)
        layer_idx: which layer to probe (0-indexed)
        position: 'last' to take the last token's representation,
                  'mean' to average over all tokens,
                  or an int for a specific token position
    Returns:
        hiddens: shape (batch, hidden_dim)
    """
    pass

def train_probe(
    hiddens_train: torch.Tensor,
    labels_train: torch.Tensor,
    max_iter: int = 1000,
    C: float = 1.0,
) -> object:
    """
    Train a logistic regression probe on hidden states.

    Args:
        hiddens_train: shape (n_train, hidden_dim), the activations
        labels_train: shape (n_train,), binary labels (0 or 1)
        max_iter: max iterations for solver
        C: inverse regularization strength (smaller = more regularized)
    Returns:
        probe: fitted sklearn LogisticRegression object
    """
    pass

def probe_accuracy(probe: object, hiddens: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Evaluate probe accuracy on a dataset.

    Args:
        probe: fitted LogisticRegression
        hiddens: shape (n, hidden_dim)
        labels: shape (n,), binary labels
    Returns:
        accuracy: float in [0, 1]
    """
    pass

def layer_probe_sweep(
    model: nn.Module,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    val_inputs: torch.Tensor,
    val_labels: torch.Tensor,
    position: int | str = 'last',
) -> dict[str, list]:
    """
    Train and evaluate a probe at every layer of the model.

    Args:
        model: transformer with model.layers attribute
        train_inputs: input_ids for training, shape (n_train, seq_len)
        train_labels: binary labels for training, shape (n_train,)
        val_inputs: input_ids for validation, shape (n_val, seq_len)
        val_labels: binary labels for validation, shape (n_val,)
        position: token position to probe ('last', 'mean', or int)
    Returns:
        dict with 'layer_idx', 'val_accuracy', 'train_accuracy' — one entry per layer
    """
    pass
```

## Answer

**Key concepts:**
1. A linear probe is a simple classifier (logistic regression) trained on frozen model activations
2. **High probe accuracy** = the model linearly encodes this concept at this layer
3. **Low probe accuracy** = the concept is either not encoded, or encoded nonlinearly
4. Probing is **diagnostic** — it tells you what's in the model's representations but not how the model uses it
5. The probe should be simple (linear): if you need a deep network to probe, you're essentially training a new model, not probing
6. **Layer sweep** reveals how concepts develop through the network — early layers encode syntax/surface form, later layers encode semantics/intent

**Reference implementation:**
```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression

def extract_hidden_states(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int,
    position: int | str = 'last',
) -> torch.Tensor:
    """Extract hidden states from a specific layer."""
    captured = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        captured['hidden'] = hidden.detach()

    hook = model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        hook.remove()

    hidden = captured['hidden']  # (batch, seq_len, hidden_dim)

    if position == 'last':
        return hidden[:, -1, :]    # Last token's representation
    elif position == 'mean':
        return hidden.mean(dim=1)  # Average over sequence
    else:
        return hidden[:, position, :]  # Specific position

def train_probe(
    hiddens_train: torch.Tensor,
    labels_train: torch.Tensor,
    max_iter: int = 1000,
    C: float = 1.0,
) -> LogisticRegression:
    """Fit logistic regression on hidden states."""
    X = hiddens_train.cpu().numpy()
    y = labels_train.cpu().numpy()

    probe = LogisticRegression(max_iter=max_iter, C=C, random_state=1)
    probe.fit(X, y)
    return probe

def probe_accuracy(
    probe: LogisticRegression,
    hiddens: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Evaluate probe accuracy."""
    X = hiddens.cpu().numpy()
    y = labels.cpu().numpy()
    preds = probe.predict(X)
    return (preds == y).mean()

def layer_probe_sweep(
    model: nn.Module,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    val_inputs: torch.Tensor,
    val_labels: torch.Tensor,
    position: int | str = 'last',
) -> dict[str, list]:
    """Train probes at every layer and return accuracies."""
    n_layers = len(model.layers)
    results = {'layer_idx': [], 'train_accuracy': [], 'val_accuracy': []}

    for layer_idx in range(n_layers):
        train_hiddens = extract_hidden_states(model, train_inputs, layer_idx, position)
        val_hiddens = extract_hidden_states(model, val_inputs, layer_idx, position)

        probe = train_probe(train_hiddens, train_labels)

        train_acc = probe_accuracy(probe, train_hiddens, train_labels)
        val_acc = probe_accuracy(probe, val_hiddens, val_labels)

        results['layer_idx'].append(layer_idx)
        results['train_accuracy'].append(train_acc)
        results['val_accuracy'].append(val_acc)

    return results
```

**Testing:**
```python
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(1)
np.random.seed(1)

# Build a toy transformer for testing
class FeedForward(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d * 4), nn.ReLU(), nn.Linear(d * 4, d))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class Block(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.ff = FeedForward(d)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(self.norm(x))

class ToyLM(nn.Module):
    def __init__(self, vocab: int = 50, d: int = 64, n_layers: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([Block(d) for _ in range(n_layers)])
        self.head = nn.Linear(d, vocab)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)

model = ToyLM()
model.eval()

# Test 1: Extract hidden states shapes
print("=" * 70)
print("TEST 1: Extract Hidden States")
print("=" * 70)
input_ids = torch.randint(0, 50, (8, 12))

h_last = extract_hidden_states(model, input_ids, layer_idx=1, position='last')
h_mean = extract_hidden_states(model, input_ids, layer_idx=1, position='mean')
h_pos0 = extract_hidden_states(model, input_ids, layer_idx=1, position=0)

print(f"Input shape: {input_ids.shape}")
print(f"Last token:  {h_last.shape}  (should be (8, 64))")
print(f"Mean pool:   {h_mean.shape}  (should be (8, 64))")
print(f"Position 0:  {h_pos0.shape}  (should be (8, 64))")

# Test 2: Probe on a learnable concept
print("\n" + "=" * 70)
print("TEST 2: Probe Detects a Linearly Encoded Concept")
print("=" * 70)

# Concept: label=1 if first token >= 25, else 0
# We bias the model's embeddings so this is detectable
with torch.no_grad():
    # Create a detectable linear signal in the embedding
    for i in range(25, 50):
        model.embed.weight[i, 0] += 3.0  # Large positive component for class 1

model.eval()

def make_dataset(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Class 1: starts with high token (>=25), class 0: low token (<25)
    tokens_class1 = torch.randint(25, 50, (n // 2, 1))
    tokens_class0 = torch.randint(0, 25, (n // 2, 1))
    rest1 = torch.randint(0, 50, (n // 2, 11))
    rest0 = torch.randint(0, 50, (n // 2, 11))
    X1 = torch.cat([tokens_class1, rest1], dim=1)
    X0 = torch.cat([tokens_class0, rest0], dim=1)
    X = torch.cat([X1, X0], dim=0)
    y = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)]).long()
    perm = torch.randperm(n)
    return X[perm], y[perm]

X_train, y_train = make_dataset(200)
X_val, y_val = make_dataset(100)

# Train probe at last layer
hiddens_train = extract_hidden_states(model, X_train, layer_idx=3, position=0)  # probe first token
hiddens_val = extract_hidden_states(model, X_val, layer_idx=3, position=0)

probe = train_probe(hiddens_train, y_train)
train_acc = probe_accuracy(probe, hiddens_train, y_train)
val_acc = probe_accuracy(probe, hiddens_val, y_val)

print(f"Train accuracy: {train_acc:.4f}")
print(f"Val accuracy:   {val_acc:.4f}")
print(f"Better than chance (0.5): {val_acc > 0.6}")

# Test 3: Layer sweep
print("\n" + "=" * 70)
print("TEST 3: Layer Probe Sweep")
print("=" * 70)

results = layer_probe_sweep(model, X_train, y_train, X_val, y_val, position=0)

print(f"{'Layer':>6} | {'Train Acc':>10} | {'Val Acc':>8}")
print("-" * 32)
for i, (layer, train_a, val_a) in enumerate(zip(
    results['layer_idx'], results['train_accuracy'], results['val_accuracy']
)):
    marker = " <-- best" if val_a == max(results['val_accuracy']) else ""
    print(f"{layer:>6} | {train_a:>10.4f} | {val_a:>8.4f}{marker}")

best_layer = results['layer_idx'][np.argmax(results['val_accuracy'])]
print(f"\nBest layer: {best_layer}")
print(f"Concept is most linearly decodable at layer {best_layer}")

# Test 4: Probe coefficient reveals the concept direction
print("\n" + "=" * 70)
print("TEST 4: Probe Weights as Concept Direction")
print("=" * 70)

hiddens = extract_hidden_states(model, X_train, layer_idx=best_layer, position=0)
probe = train_probe(hiddens, y_train)

coef = torch.tensor(probe.coef_[0])  # (hidden_dim,) — the concept direction
print(f"Probe weight vector shape: {coef.shape}")
print(f"Highest-weight dimension: {coef.abs().argmax().item()}")
print(f"Weight at dim 0 (where we added signal): {coef[0].item():.4f}")
print(f"Dim 0 has large weight: {coef[0].abs().item() > coef[1:].abs().mean().item()}")
print("(Probe correctly finds the concept direction we embedded)")

# Test 5: Random concept is not detectable
print("\n" + "=" * 70)
print("TEST 5: Random Labels Are Not Decodable")
print("=" * 70)

random_labels = torch.randint(0, 2, (len(X_train),)).long()
random_val_labels = torch.randint(0, 2, (len(X_val),)).long()

sweep_random = layer_probe_sweep(model, X_train, random_labels, X_val, random_val_labels)
best_random_val_acc = max(sweep_random['val_accuracy'])
print(f"Best val accuracy on random labels: {best_random_val_acc:.4f}")
print(f"Near chance (0.5): {abs(best_random_val_acc - 0.5) < 0.15}")
```

**Common mistakes:**
1. Forgetting to call `hook.remove()` — unregistered hooks slow down all subsequent forward passes
2. Probing on training data only and reporting training accuracy as the probe accuracy — always use a held-out set
3. Using a nonlinear probe (MLP) — this no longer tests *linear* decodability; you're just training a new model
4. Not detaching activations before passing to sklearn — sklearn can't handle tensors with grad
5. Probing at the wrong granularity: `'last'` vs `'mean'` vs a specific position can give very different results depending on the task
6. Confusing high probe accuracy with "the model uses this concept" — probing only tells you the concept is *encoded*, not how it's used computationally

## Follow-up Questions
- What's the difference between probing and mechanistic interpretability?
- Why must the probe be linear? What does it mean if a nonlinear probe does much better?
- How does probing relate to the "linear representation hypothesis"?
- If a probe achieves 95% accuracy, does that mean the model "knows" the concept? What are the limits of this claim?
- How would you use probing to study whether a model represents its own uncertainty?
- What is the relationship between probing and activation steering?
- How does probing a safety classifier differ from probing a language model?
