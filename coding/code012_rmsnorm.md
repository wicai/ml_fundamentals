# Implement RMSNorm (Llama)

**Category:** coding
**Difficulty:** 2
**Tags:** coding, normalization, llama, modern-architectures

## Question
Implement RMSNorm as used in Llama models.

RMSNorm is a simpler alternative to LayerNorm that:
- Only normalizes by RMS (no mean centering)
- Has fewer operations (faster)
- Only uses scale parameter (no shift)

**Function signature:**
```python
class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        """
        Args:
            normalized_shape: int, dimension to normalize
            eps: float, epsilon for numerical stability
        """
        pass

    def forward(self, x):
        """
        Args:
            x: tensor of shape (..., normalized_shape)
        Returns:
            normalized tensor of same shape as x
        """
        pass
```

## Answer

**Key concepts:**
1. Compute RMS: sqrt(mean(x^2))
2. Normalize by RMS
3. Scale with learnable parameter
4. No mean subtraction, no bias

**Reference implementation:**
```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        # Compute RMS
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize
        x_norm = x / rms

        # Scale
        return self.weight * x_norm

# Alternative implementation (more efficient)
class RMSNorm_v2(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        # Equivalent but slightly different computation
        # rsqrt is more efficient than 1/sqrt
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm
```

**Testing:**
```python
# Test
batch_size, seq_len, d_model = 2, 10, 512
x = torch.randn(batch_size, seq_len, d_model)

# Your implementation
rms_norm = RMSNorm(d_model)
out = rms_norm(x)

print(f"Output shape: {out.shape}")
print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")
print(f"Output mean: {out.mean():.4f}, std: {out.std():.4f}")

# Check that RMS is normalized
rms_after = torch.sqrt(torch.mean(out ** 2, dim=-1))
print(f"RMS after normalization (should be ~1): {rms_after.mean():.4f}")

# Compare v1 vs v2
rms_norm_v2 = RMSNorm_v2(d_model)
rms_norm_v2.weight.data = rms_norm.weight.data.clone()

out_v2 = rms_norm_v2(x)
print(f"V1 vs V2 max diff: {(out - out_v2).abs().max().item():.2e}")

# Comparison with LayerNorm
ln = nn.LayerNorm(d_model)
ln.weight.data = rms_norm.weight.data.clone()
ln.bias.data.zero_()

out_ln = ln(x)
print(f"\nRMSNorm vs LayerNorm (should differ):")
print(f"Max diff: {(out - out_ln).abs().max().item():.4f}")
```

**Why RMSNorm is faster:**
```python
# Operations comparison:
# LayerNorm: mean, var, subtract, divide, scale, shift
# RMSNorm:   mean(x^2), divide, scale

# Benchmark
import time

x_large = torch.randn(64, 512, 4096).cuda()
rms = RMSNorm(4096).cuda()
ln = nn.LayerNorm(4096).cuda()

# RMSNorm
start = time.time()
for _ in range(100):
    _ = rms(x_large)
torch.cuda.synchronize()
time_rms = time.time() - start

# LayerNorm
start = time.time()
for _ in range(100):
    _ = ln(x_large)
torch.cuda.synchronize()
time_ln = time.time() - start

print(f"RMSNorm: {time_rms:.4f}s")
print(f"LayerNorm: {time_ln:.4f}s")
print(f"Speedup: {time_ln / time_rms:.2f}x")
```

**Common mistakes:**
1. ❌ Subtracting mean (that's LayerNorm, not RMSNorm)
2. ❌ Including bias parameter
3. ❌ Using sqrt instead of rsqrt (less efficient)
4. ❌ Wrong epsilon placement

## Follow-up Questions
- Why is RMSNorm simpler than LayerNorm?
- What are the tradeoffs of not centering by mean?
- Which modern LLMs use RMSNorm? (Llama, PaLM, etc.)
