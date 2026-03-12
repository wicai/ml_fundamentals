# Implement Activation Steering

**Category:** coding
**Difficulty:** 4
**Tags:** coding, interpretability, steering-vectors, activation-engineering, safety

## Question

Implement activation steering (also called representation engineering) — a technique for directly modifying model behavior by adding vectors to the residual stream at inference time.

**How it works:**
1. Collect activations from a layer while running "positive" prompts (e.g., prompts about being helpful) and "negative" prompts (e.g., prompts about being harmful)
2. The **steering vector** is the mean difference: `mean(positive_activations) - mean(negative_activations)`
3. At inference, add `alpha * steering_vector` to the residual stream to steer the model's behavior

This is used in safety research to: study model representations of concepts (honesty, harm, emotions), induce or suppress specific behaviors without fine-tuning, and understand what concepts models encode and where.

Your implementation should include:
1. **`get_layer_activations`**: Extract residual stream activations from a specific layer via hooks
2. **`compute_steering_vector`**: Find the mean-difference direction between two sets of prompts
3. **`hooked_forward`**: Run inference with a steering vector added at a specific layer

**Function signature:**
```python
def get_layer_activations(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """
    Extract residual stream activations at a specific transformer layer.

    Uses a forward hook to intercept the output of the specified layer.
    Returns the mean-pooled activation across the sequence dimension.

    Args:
        model: transformer model where model.layers[i] is the i-th transformer block
        input_ids: shape (batch, seq_len)
        layer_idx: which layer to hook (0-indexed)
    Returns:
        activations: mean-pooled over sequence, shape (batch, hidden_dim)
    """
    pass

def compute_steering_vector(
    model: nn.Module,
    positive_inputs: torch.Tensor,
    negative_inputs: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """
    Compute a steering vector as the mean difference in activations.

    steering_vector = mean(activations(positive)) - mean(activations(negative))

    Args:
        model: transformer model
        positive_inputs: input_ids for "positive" concept prompts, shape (n_pos, seq_len)
        negative_inputs: input_ids for "negative" concept prompts, shape (n_neg, seq_len)
        layer_idx: layer to extract activations from
    Returns:
        steering_vector: shape (hidden_dim,)
    """
    pass

def hooked_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    steering_vector: torch.Tensor,
    layer_idx: int,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Run a forward pass, adding alpha * steering_vector to the residual stream
    at the specified layer.

    Args:
        model: transformer model
        input_ids: shape (batch, seq_len)
        steering_vector: shape (hidden_dim,)
        layer_idx: which layer to intervene at
        alpha: steering strength. Positive = steer toward positive concept.
               Negative = steer toward negative concept.
    Returns:
        logits: shape (batch, seq_len, vocab_size)
    """
    pass
```

## Answer

**Key concepts:**
1. Transformer residual stream: each layer adds its output to a running "residual" — we intervene on this stream
2. PyTorch hooks let you intercept layer inputs/outputs without modifying model code
3. The mean-difference vector captures the "direction" of a concept in activation space
4. Adding this vector shifts the model's internal representation toward the concept
5. Alpha controls strength: too large causes incoherence, too small has no effect
6. This technique reveals that models encode concepts as linear directions in activation space (the linear representation hypothesis)

**Reference implementation:**
```python
import torch
import torch.nn as nn
from contextlib import contextmanager

def get_layer_activations(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Hook into the specified layer and capture mean-pooled activations."""
    captured = {}

    def hook_fn(module, input, output):
        # output is typically (batch, seq_len, hidden_dim) for a transformer block
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        # Mean-pool over sequence dimension
        captured['activations'] = hidden_states.mean(dim=1)  # (batch, hidden_dim)

    # Register hook on the specified layer
    hook = model.layers[layer_idx].register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        hook.remove()

    return captured['activations']

def compute_steering_vector(
    model: nn.Module,
    positive_inputs: torch.Tensor,
    negative_inputs: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """steering_vector = mean(pos_activations) - mean(neg_activations)."""
    pos_acts = get_layer_activations(model, positive_inputs, layer_idx)  # (n_pos, hidden)
    neg_acts = get_layer_activations(model, negative_inputs, layer_idx)  # (n_neg, hidden)

    return pos_acts.mean(dim=0) - neg_acts.mean(dim=0)  # (hidden_dim,)

def hooked_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    steering_vector: torch.Tensor,
    layer_idx: int,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Forward pass with steering vector added to residual stream."""
    def steering_hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
            # Add steering vector to all token positions
            steered = hidden_states + alpha * steering_vector.to(hidden_states.device)
            return (steered,) + rest
        else:
            return output + alpha * steering_vector.to(output.device)

    hook = model.layers[layer_idx].register_forward_hook(steering_hook)

    try:
        with torch.no_grad():
            logits = model(input_ids)
    finally:
        hook.remove()

    return logits
```

**Testing:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# Build a minimal transformer-like model for testing
class FeedForward(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(self.norm(x))

class ToyLM(nn.Module):
    def __init__(self, vocab_size: int = 100, d_model: int = 64, n_layers: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

model = ToyLM()
model.eval()

# Test 1: Get layer activations
print("=" * 70)
print("TEST 1: Get Layer Activations")
print("=" * 70)

input_ids = torch.randint(0, 100, (4, 10))
acts = get_layer_activations(model, input_ids, layer_idx=2)
print(f"Input shape: {input_ids.shape}")
print(f"Activations shape: {acts.shape}  (should be (4, 64))")
print(f"Mean-pooled over sequence: correct")

# Test 2: Activations differ across layers
print("\n" + "=" * 70)
print("TEST 2: Activations Differ Across Layers")
print("=" * 70)

acts_layer0 = get_layer_activations(model, input_ids, layer_idx=0)
acts_layer3 = get_layer_activations(model, input_ids, layer_idx=3)
print(f"Layer 0 act mean: {acts_layer0.mean().item():.4f}")
print(f"Layer 3 act mean: {acts_layer3.mean().item():.4f}")
print(f"Layers produce different activations: {not torch.allclose(acts_layer0, acts_layer3)}")

# Test 3: Compute steering vector
print("\n" + "=" * 70)
print("TEST 3: Compute Steering Vector")
print("=" * 70)

# Simulate: positive prompts use tokens from range [0, 49], negative from [50, 99]
pos_inputs = torch.randint(0, 50, (8, 10))   # "positive" concept
neg_inputs = torch.randint(50, 100, (8, 10))  # "negative" concept

sv = compute_steering_vector(model, pos_inputs, neg_inputs, layer_idx=2)
print(f"Steering vector shape: {sv.shape}  (should be (64,))")
print(f"Steering vector norm: {sv.norm().item():.4f}")
print(f"Not zero vector: {sv.norm().item() > 0}")

# Test 4: Hooked forward changes outputs
print("\n" + "=" * 70)
print("TEST 4: Hooked Forward Changes Model Output")
print("=" * 70)

test_input = torch.randint(0, 100, (2, 8))

logits_baseline = model(test_input)
logits_steered = hooked_forward(model, test_input, sv, layer_idx=2, alpha=1.0)

print(f"Output shape: {logits_steered.shape}")
print(f"Steered differs from baseline: {not torch.allclose(logits_baseline, logits_steered)}")
max_diff = (logits_steered - logits_baseline).abs().max().item()
print(f"Max logit difference: {max_diff:.4f}")

# Test 5: Alpha controls intervention strength
print("\n" + "=" * 70)
print("TEST 5: Alpha Controls Steering Strength")
print("=" * 70)

for alpha in [0.0, 0.5, 1.0, 5.0]:
    steered = hooked_forward(model, test_input, sv, layer_idx=2, alpha=alpha)
    diff = (steered - logits_baseline).abs().mean().item()
    print(f"  alpha={alpha:.1f}: mean abs logit diff = {diff:.4f}")

print("alpha=0.0 should give zero diff (no steering)")

# Test 6: Negative alpha steers toward negative concept
print("\n" + "=" * 70)
print("TEST 6: Negative Alpha Steers Toward Negative Concept")
print("=" * 70)

steered_pos = hooked_forward(model, test_input, sv, layer_idx=2, alpha=5.0)
steered_neg = hooked_forward(model, test_input, sv, layer_idx=2, alpha=-5.0)

# Positive-alpha output should be closer to activations from positive prompts
pos_acts_baseline = get_layer_activations(model, pos_inputs[:2], layer_idx=2)

diff_pos_steered_pos = (steered_pos - logits_baseline).mean().item()
diff_neg_steered_neg = (steered_neg - logits_baseline).mean().item()
print(f"Positive steering shifts logits by: {diff_pos_steered_pos:+.4f}")
print(f"Negative steering shifts logits by: {diff_neg_steered_neg:+.4f}")
print(f"Opposite signs: {diff_pos_steered_pos * diff_neg_steered_neg < 0}")
```

**Common mistakes:**
1. Forgetting to call `hook.remove()` — hooks persist and will affect all future forward passes
2. Not handling the case where layer output is a tuple (attention layers often return `(hidden_states, attention_weights, ...)`)
3. Adding the steering vector to the **input** of the layer instead of the **output** (post-layer residual)
4. Not using `torch.no_grad()` when collecting activations for steering vector computation
5. Using `last_token` instead of `mean` pooling — mean is more robust for computing concept directions
6. Setting alpha too high: the model's internal representations become incoherent, and output degrades

## Follow-up Questions
- Why does the mean-difference vector work as a steering direction? (Linear representation hypothesis)
- How do you find the right layer to intervene on for a given concept?
- What is the difference between activation steering and fine-tuning? When would you prefer each?
- How does this relate to "representation engineering" (Zou et al. 2023)?
- What are the limitations of activation steering for safety interventions?
- How would you measure whether a steering intervention is working? (Beyond just "outputs changed")
- What is the relationship between steering vectors and the directions found by sparse autoencoders?
