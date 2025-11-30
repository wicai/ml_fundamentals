# Backpropagation

**Category:** foundations
**Difficulty:** 4
**Tags:** optimization, fundamentals, gradients

## Question
Explain backpropagation and why it's efficient for training neural networks.

## Answer
**Backpropagation**: Efficient algorithm for computing gradients using the chain rule.

**The problem:**
Need ∂L/∂w for all parameters w to update them via gradient descent.

**Naive approach:**
Compute each ∂L/∂w_i independently → O(n) forward passes for n parameters.

**Backprop insight:**
Use chain rule + dynamic programming to compute all gradients in single backward pass.

**Forward pass** (build computation graph):
```
z = Wx + b
a = ReLU(z)
L = loss(a, y)
```

**Backward pass** (chain rule):
```
∂L/∂L = 1                          # Base case
∂L/∂a = ∂L/∂L * ∂loss/∂a           # Chain rule
∂L/∂z = ∂L/∂a * ∂ReLU/∂z           # Element-wise
∂L/∂W = ∂L/∂z * ∂z/∂W = ∂L/∂z * x^T  # Matrix gradient
∂L/∂b = ∂L/∂z * ∂z/∂b = ∂L/∂z      # Bias gradient
∂L/∂x = W^T * ∂L/∂z                 # Propagate to previous layer
```

**Computational graph perspective:**
- Forward: Compute output and cache intermediates
- Backward: Traverse graph in reverse, accumulate gradients

**Efficiency:**
- **Time**: O(1) cost per edge in computation graph
- **Space**: Store activations for backward pass (main memory cost)

**Key tricks:**

1. **Gradient accumulation**: Sum gradients if parameter used multiple times
2. **Activation checkpointing**: Trade compute for memory (recompute activations)
3. **Automatic differentiation**: Frameworks handle this automatically (PyTorch, JAX)

**Gotchas:**
- **Vanishing gradients**: ∂L/∂z → 0 in deep nets (sigmoid, tanh)
- **Exploding gradients**: ∂L/∂z → ∞ (fixed by gradient clipping)

## Follow-up Questions
- What's the memory cost of backprop?
- How does automatic differentiation work?
- What causes vanishing/exploding gradients?
