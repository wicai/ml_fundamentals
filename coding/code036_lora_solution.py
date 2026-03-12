# Implement LoRA (Low-Rank Adaptation)
# ====================================================================
#
# Implement LoRA (Low-Rank Adaptation) — the standard technique for parameter-efficient fine-tuning. Instead of updating the full weight matrix W, LoRA adds a low-rank update: W' = W + BA, where B and A are small matrices.
# 
# Your implementation should include:
# 1. **`LoRALinear`**: A drop-in replacement for `nn.Linear` that adds LoRA parameters
# 2. **`apply_lora`**: Inject LoRA into all linear layers of an existing model
# 3. **`merge_lora`**: Merge LoRA weights back into the base model for efficient inference
# 
# **Function signature:**
#
# ====================================================================
import torch 
from torch import nn
import torch.nn.functional as F
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
        super().__init__()
        self.base_linear = base_linear
        for param in self.base_linear.parameters():
            param.requires_grad=False
        self.rank = rank
        self.alpha = alpha
        # get dims of base_linear (dim_out, dim_in)
        out_features = self.base_linear.out_features
        in_features = self.base_linear.in_features        
        # B is (dim_out, r)
        self.B = nn.Parameter(torch.zeros(out_features, self.rank))
        # A is (r, dim_in)
        self.A = nn.Parameter(torch.randn(self.rank, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base output + scaled LoRA output."""
        # lienar layers are x @ the stored weights transposed 
        # x@W^T normally
        # x @ (W + BA)^T with lora = X @ W^T + x @ A^T @ B^T
        return self.base_linear(x) + (x @ self.A.T @ self.B.T * self.alpha/self.rank)

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into base weight and return a standard nn.Linear."""
        w = self.base_linear.weight
        combined = w + self.B @ self.A * (self.alpha/self.rank)        
        combined_layer = nn.Linear(
            self.base_linear.in_features,
            self.base_linear.out_features,
            bias = (self.base_linear.bias is not None)
        )
        with torch.no_grad():
            combined_layer.weight.data = combined
            if self.base_linear.bias is not None:
                combined_layer.bias.data = self.base_linear.bias
        return combined_layer

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
    for name, module in model.named_modules():
        if target_modules is not None:
            if not(any(t in name for t in target_modules)): # no matches
                continue # skip this one
        if not isinstance(module, nn.Linear):
            continue
        # if we get here, it's a nn.Linear that passes the filter if it is one    
        # here, the name is something like a or a.b or a.b.c
        parent = model
        # if there's . keep traversing parent until it's at the penultimate dots        
        to_traverse = name.split('.')[:-1]
        for m in to_traverse:
            parent = getattr(parent, m)
        setattr(
            parent,
            name.split('.')[-1], 
            LoRALinear(module, rank, alpha)
        )
    return model

def count_parameters(model: nn.Module) -> dict[str, int]:
    """
    Count total, trainable, and frozen parameters.

    Returns:
        dict with 'total', 'trainable', 'frozen'
    """
    return_dict = {
        'total':0,
        'trainable':0,
        'frozen':0
    }
    for parameter in model.parameters():
        n_params = parameter.numel()
        return_dict['total'] += n_params
        if parameter.requires_grad:
            return_dict['trainable'] += n_params
        else:
            return_dict['frozen'] += n_params
    return return_dict

if __name__ == "__main__":
    torch.manual_seed(1)

    print("=" * 70)
    print("TEST 1: LoRALinear Basics")
    base = nn.Linear(64, 128)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(2, 64)
    base_out = base(x)
    lora_out = lora(x)
    print(f"Initial output matches base: {torch.allclose(base_out, lora_out, atol=1e-6)}")
    print(f"Base weight frozen: {not base.weight.requires_grad}")

    print("\n" + "=" * 70)
    print("TEST 2: Merge LoRA Into Base")
    lora.A.data.normal_(std=0.1)
    lora.B.data.normal_(std=0.1)
    x = torch.randn(4, 64)
    lora_out = lora(x)
    merged = lora.merge()
    merged_out = merged(x)
    print(f"LoRA output matches merged: {torch.allclose(lora_out, merged_out, atol=1e-5)}")

    print("\n" + "=" * 70)
    print("TEST 3: Apply LoRA to Model")
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
    apply_lora(model, rank=4, alpha=8.0)
    params = count_parameters(model)
    print(f"Params: {params}")
    print(f"Trainable ratio: {params['trainable'] / params['total']:.2%}")
    out = model(torch.randn(2, 32))
    print(f"Output shape: {out.shape} (should be (2, 10))")

    print("\n" + "=" * 70)
    print("TEST 4: Selective LoRA")
    class SimpleTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
        def forward(self, x): return self.v_proj(x)
    model = SimpleTransformer()
    apply_lora(model, rank=4, target_modules=['q_proj', 'v_proj'])
    print(f"q_proj is LoRA: {isinstance(model.q_proj, LoRALinear)}")
    print(f"k_proj is LoRA: {isinstance(model.k_proj, LoRALinear)} (should be False)")
    print(f"v_proj is LoRA: {isinstance(model.v_proj, LoRALinear)}")
