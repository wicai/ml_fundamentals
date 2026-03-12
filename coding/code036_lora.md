# Implement LoRA (Low-Rank Adaptation)

**Category:** coding
**Difficulty:** 3
**Tags:** coding, lora, fine-tuning, peft, safety

## Question
Implement LoRA (Low-Rank Adaptation) — the standard technique for parameter-efficient fine-tuning. Instead of updating the full weight matrix W, LoRA adds a low-rank update: W' = W + BA, where B and A are small matrices.

Your implementation should include:
1. **`LoRALinear`**: A drop-in replacement for `nn.Linear` that adds LoRA parameters
2. **`apply_lora`**: Inject LoRA into all linear layers of an existing model
3. **`merge_lora`**: Merge LoRA weights back into the base model for efficient inference

**Function signature:**
```python
class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    output = x @ (W + B @ A)^T + bias
           = x @ W^T + x @ A^T @ B^T + bias
           = original_output + lora_output

    The base weight W is frozen. Only A and B are trained.
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        """
        Args:
            base_linear: the original nn.Linear layer to adapt
            rank: LoRA rank (r). Lower = fewer params, higher = more expressive.
            alpha: scaling factor. The LoRA output is scaled by alpha/rank.
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base output + scaled LoRA output."""
        pass

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into base weight and return a standard nn.Linear."""
        pass

def apply_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0, target_modules: list[str] | None = None) -> nn.Module:
    """
    Replace linear layers in the model with LoRA-adapted versions.
    Freeze all base parameters, only LoRA params are trainable.

    Args:
        model: the base model
        rank: LoRA rank
        alpha: LoRA scaling factor
        target_modules: list of module name substrings to target (e.g., ['q_proj', 'v_proj']).
                       If None, apply to all nn.Linear layers.
    Returns:
        model with LoRA layers (modified in-place)
    """
    pass

def count_parameters(model: nn.Module) -> dict[str, int]:
    """
    Count total, trainable, and frozen parameters.

    Returns:
        dict with 'total', 'trainable', 'frozen'
    """
    pass
```

## Answer

**Key concepts:**
1. W' = W + (alpha/r) * B @ A, where A is (r, in_features), B is (out_features, r)
2. A is initialized with random Gaussian, B is initialized to zeros → initial LoRA output is zero
3. Only A and B are trainable; W is frozen
4. alpha/r is the scaling factor — controls the magnitude of the LoRA update
5. At inference, merge B@A into W for zero overhead

**Reference implementation:**
```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()

        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # LoRA matrices
        # A: down-projection (in_features → rank)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * (1 / math.sqrt(rank)))
        # B: up-projection (rank → out_features), initialized to zero
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Freeze the base weights
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base output (frozen)
        base_out = self.base_linear(x)

        # LoRA output: x @ A^T @ B^T * scaling
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

        return base_out + lora_out

    def merge(self) -> nn.Linear:
        """Merge LoRA into base weight: W_merged = W + scaling * B @ A"""
        merged = nn.Linear(
            self.base_linear.in_features,
            self.base_linear.out_features,
            bias=self.base_linear.bias is not None,
        )
        with torch.no_grad():
            merged.weight.copy_(
                self.base_linear.weight + self.scaling * (self.lora_B @ self.lora_A)
            )
            if self.base_linear.bias is not None:
                merged.bias.copy_(self.base_linear.bias)
        return merged

def apply_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0, target_modules: list[str] | None = None) -> nn.Module:
    """Replace matching linear layers with LoRA versions."""
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if target_modules is not None:
            if not any(target in name for target in target_modules):
                continue

        # Replace the module
        lora_layer = LoRALinear(module, rank=rank, alpha=alpha)

        # Navigate to parent and replace
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora_layer)

    return model

def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total, trainable, and frozen parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
    }
```

**Testing:**
```python
import torch
import torch.nn as nn

torch.manual_seed(1)

# Test 1: LoRALinear basic behavior
print("=" * 70)
print("TEST 1: LoRALinear Basics")
print("=" * 70)
base = nn.Linear(64, 128)
lora = LoRALinear(base, rank=4, alpha=8.0)

x = torch.randn(2, 64)

# At initialization, B=0, so LoRA output should match base
base_out = base(x)
lora_out = lora(x)
print(f"Initial output matches base: {torch.allclose(base_out, lora_out, atol=1e-6)}")
print(f"Base weight frozen: {not base.weight.requires_grad}")
print(f"LoRA A trainable: {lora.lora_A.requires_grad}")
print(f"LoRA B trainable: {lora.lora_B.requires_grad}")

# After some training, outputs diverge
lora.lora_B.data.normal_()
lora_out_after = lora(x)
print(f"After updating B, outputs differ: {not torch.allclose(base_out, lora_out_after)}")

# Test 2: Parameter count
print("\n" + "=" * 70)
print("TEST 2: Parameter Efficiency")
print("=" * 70)
rank = 4
params_base = 64 * 128 + 128  # weight + bias
params_lora = rank * 64 + 128 * rank  # A + B
print(f"Base params: {params_base}")
print(f"LoRA params (rank={rank}): {params_lora}")
print(f"Ratio: {params_lora / params_base:.2%}")

# Test 3: Merge
print("\n" + "=" * 70)
print("TEST 3: Merge LoRA Into Base")
print("=" * 70)
lora.lora_A.data.normal_(std=0.1)
lora.lora_B.data.normal_(std=0.1)

x = torch.randn(4, 64)
lora_out = lora(x)
merged = lora.merge()
merged_out = merged(x)
print(f"LoRA output matches merged: {torch.allclose(lora_out, merged_out, atol=1e-5)}")
print(f"Merged is plain nn.Linear: {type(merged).__name__}")

# Test 4: Apply LoRA to a model
print("\n" + "=" * 70)
print("TEST 4: Apply LoRA to Model")
print("=" * 70)
model = nn.Sequential(
    nn.Linear(32, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)

params_before = count_parameters(model)
print(f"Before LoRA: {params_before}")

apply_lora(model, rank=4, alpha=8.0)
params_after = count_parameters(model)
print(f"After LoRA:  {params_after}")
print(f"Trainable reduced: {params_after['trainable']} vs {params_before['trainable']}")
print(f"Trainable ratio: {params_after['trainable'] / params_after['total']:.2%}")

# Verify forward pass still works
x = torch.randn(2, 32)
out = model(x)
print(f"Output shape: {out.shape} (should be (2, 10))")

# Test 5: Selective LoRA (only some layers)
print("\n" + "=" * 70)
print("TEST 5: Selective LoRA (target_modules)")
print("=" * 70)

class SimpleTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.out_proj = nn.Linear(64, 64)
        self.ffn = nn.Linear(64, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.out_proj(self.v_proj(x)))

model = SimpleTransformer()
apply_lora(model, rank=4, target_modules=['q_proj', 'v_proj'])

print(f"q_proj is LoRA: {isinstance(model.q_proj, LoRALinear)}")
print(f"k_proj is LoRA: {isinstance(model.k_proj, LoRALinear)} (should be False)")
print(f"v_proj is LoRA: {isinstance(model.v_proj, LoRALinear)}")
print(f"ffn is LoRA: {isinstance(model.ffn, LoRALinear)} (should be False)")
```

**Common mistakes:**
1. Initializing both A and B randomly (should init B=0 so initial output is unchanged)
2. Forgetting the scaling factor alpha/r
3. Not freezing base weights (then you're training everything, not just LoRA)
4. Wrong matrix dimensions: A is (r, in_features), B is (out_features, r)
5. Forgetting to handle bias when merging

## Follow-up Questions
- Why initialize B to zero and not A?
- How do you choose the rank r? What are typical values?
- What is QLoRA and how does it extend LoRA?
- Why apply LoRA to attention projections (Q, V) and not feedforward layers?
- How does LoRA compare to full fine-tuning in terms of quality?
