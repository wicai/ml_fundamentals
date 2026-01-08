# Implement Batch Normalization

**Category:** coding
**Difficulty:** 3
**Tags:** coding, normalization, training

## Question
Implement batch normalization from scratch in PyTorch.

Your implementation should:
- Normalize across the batch dimension
- Support training and evaluation modes
- Track running statistics (running mean/var)
- Include learnable affine parameters

**Function signature:**
```python
class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        Args:
            num_features: number of features (C from (N, C))
            eps: epsilon for numerical stability
            momentum: momentum for running statistics
        """
        pass

    def forward(self, x):
        """
        Args:
            x: shape (batch_size, num_features)
        Returns:
            normalized tensor of same shape
        """
        pass
```

## Answer

**Key concepts:**
1. Training: normalize using batch statistics
2. Eval: normalize using running statistics
3. Update running stats with momentum
4. Learnable scale (gamma) and shift (beta)

**Reference implementation:**
```python
import torch
import torch.nn as nn

class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Running statistics (not parameters, but part of state)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # x: (batch_size, num_features)

        if self.training:
            # Compute batch statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            # Update running statistics
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + \
                                   self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + \
                                  self.momentum * batch_var
                self.num_batches_tracked += 1

            # Normalize using batch statistics
            mean = batch_mean
            var = batch_var
        else:
            # Normalize using running statistics
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        out = self.gamma * x_norm + self.beta

        return out
```

**Testing:**
```python
# Test
batch_size, num_features = 32, 10
bn_custom = BatchNorm1d(num_features)
bn_pytorch = nn.BatchNorm1d(num_features)

# Copy parameters
bn_pytorch.weight.data = bn_custom.gamma.data
bn_pytorch.bias.data = bn_custom.beta.data

# Training mode
bn_custom.train()
bn_pytorch.train()

x = torch.randn(batch_size, num_features)
out_custom = bn_custom(x)
out_pytorch = bn_pytorch(x)

print(f"Training - Max diff: {(out_custom - out_pytorch).abs().max().item():.2e}")

# Eval mode
bn_custom.eval()
bn_pytorch.eval()

x_test = torch.randn(batch_size, num_features)
out_custom_eval = bn_custom(x_test)
out_pytorch_eval = bn_pytorch(x_test)

print(f"Eval - Max diff: {(out_custom_eval - out_pytorch_eval).abs().max().item():.2e}")
```

**Common mistakes:**
1. ❌ Using unbiased=True for variance (should be biased)
2. ❌ Not tracking running statistics
3. ❌ Wrong momentum update (should be exponential moving average)
4. ❌ Using batch stats during eval

## Follow-up Questions
- Why use biased variance in batch norm?
- How does batch size affect batch norm performance?
- When would you use LayerNorm instead of BatchNorm?
