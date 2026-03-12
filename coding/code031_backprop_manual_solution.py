# Implement Manual Backpropagation
# ====================================================================
#
# Implement forward and backward passes manually for a simple 2-layer neural network, WITHOUT using `autograd`. This builds intuition for what `.backward()` actually does.
# 
# Network: `Input → Linear → ReLU → Linear → MSE Loss`
# 
# Your implementation should:
# 1. Implement the forward pass, caching intermediate values needed for backprop
# 2. Implement the backward pass, computing gradients for all weights and biases
# 3. Verify your gradients match PyTorch's autograd
# 
# **Function signature:**
#
# ====================================================================

import torch

class ManualMLP:
    """
    2-layer MLP with manual forward and backward passes.
    Architecture: Linear(in, hidden) → ReLU → Linear(hidden, out)
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """Initialize weights (use torch tensors, but no autograd)."""
        self.w1 = torch.randn([input_dim, hidden_dim])
        self.w2 = torch.randn([hidden_dim, output_dim])
        self.b1 = torch.zeros([hidden_dim])
        self.b2 = torch.zeros([output_dim])        
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Cache intermediate values for backward.

        Args:
            x: input, shape (batch_size, input_dim)
        Returns:
            output: shape (batch_size, output_dim)
        """
        self.x = x
        self.z1 = x @ self.w1 + self.b1 # (batch_size, hidden_dim)
        self.a1 = torch.clamp(self.z1, min = 0) # (batch_size, hidden_dim)
        self.output = self.a1 @ self.w2 + self.b2 # (batch_size, output_dim)
        return self.output
        

    def backward(self, grad_output: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Backward pass. Compute gradients for all parameters.

        Args:
            grad_output: gradient of loss w.r.t. output, shape (batch_size, output_dim)
        Returns:
            dict mapping parameter names to their gradients
        """
        # dloss/doutput = grad_output
        # doutput/db2 = 1
        # doutput/dw2 = a1.transpose() @ grad_output # (hidden_dim, batch_size) @ (batch_size, output_dim)
        # doutput/da1 = self.w2T @ grad_output
        grad_b2 = grad_output.sum(dim=0) #(output)
        grad_w2 = self.a1.T @ grad_output # (hidden, output)
        grad_a1 = grad_output @ self.w2.T #(batch_size, hidden_dim)
        # dloss/dz1 = dloss/da1 * da1/dz1
        # da1/dz1 is 1 if z1 > 0 else 0
        # (z1 > 0) # (batch_size, hidden_dim)
        grad_z1 = (self.z1 > 0) * grad_a1
        #grad_b2 = grad_z1 * dz1/db2 = grad_z1 * 1
        grad_b1 = grad_z1.sum(dim=0)
        grad_w1 = self.x.T @ grad_z1
        return {
            'w1': grad_w1,
            'w2': grad_w2,
            'b1': grad_b1,
            'b2': grad_b2,
        }


def mse_loss_and_grad(predictions: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute MSE loss and its gradient w.r.t. predictions.

    Args:
        predictions: shape (batch_size, output_dim)
        targets: shape (batch_size, output_dim)
    Returns:
        loss: scalar
        grad: shape (batch_size, output_dim), gradient of loss w.r.t. predictions
    """
    mse = torch.mean((predictions-targets) ** 2)    
    grad_mse = 2 * (predictions-targets) / predictions.numel() # ()
    return (mse, grad_mse)


def test_gradients():
    """Verify manual gradients match PyTorch autograd."""
    torch.manual_seed(1)

    mlp = ManualMLP(3, 4, 2)
    x = torch.randn(5, 3)
    targets = torch.randn(5, 2)

    # --- Manual forward/backward ---
    output = mlp.forward(x)
    loss, grad_loss = mse_loss_and_grad(output, targets)
    manual_grads = mlp.backward(grad_loss)

    # --- PyTorch autograd for comparison ---
    torch.manual_seed(1)
    w1 = torch.randn(3, 4, requires_grad=True)
    w2 = torch.randn(4, 2, requires_grad=True)
    b1 = torch.zeros(4, requires_grad=True)
    b2 = torch.zeros(2, requires_grad=True)

    z1 = x @ w1 + b1
    a1 = torch.clamp(z1, min=0)
    out = a1 @ w2 + b2
    auto_loss = torch.mean((out - targets) ** 2)
    auto_loss.backward()

    print(f"Loss (manual):   {loss.item():.6f}")
    print(f"Loss (autograd): {auto_loss.item():.6f}")
    print()

    for name, param in [('w1', w1), ('w2', w2), ('b1', b1), ('b2', b2)]:
        max_diff = (manual_grads[name] - param.grad).abs().max().item()
        match = "PASS" if max_diff < 1e-6 else "FAIL"
        print(f"grad_{name}: max_diff = {max_diff:.2e}  [{match}]")


if __name__ == "__main__":
    test_gradients()
