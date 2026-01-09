# Weight Initialization

**Category:** foundations
**Difficulty:** 3
**Tags:** initialization, training, optimization

## Question
How should you initialize weights in deep networks and why does it matter?

## What to Cover
- **Set context by**: Explaining the problems bad initialization causes (vanishing/exploding gradients, symmetry breaking)
- **Must mention**: Key strategies (Xavier/Glorot, He), variance preservation principle, transformer-specific practices (Normal(0, 0.02), layer-wise scaling)
- **Show depth by**: Noting that modern techniques (layer norm, residual connections) reduce initialization sensitivity
- **Avoid**: Only describing the formulas without explaining *why* proper variance matters

## Answer
**Goal**: Start with weights that allow gradients to flow without vanishing/exploding.

**Bad initialization:**
```
All zeros: Neurons compute same thing, symmetry never broken
All same value: Same problem
Too large: Activations explode → NaN
Too small: Activations vanish → no learning
```

**Good initialization strategies:**

**1. Xavier/Glorot (Tanh, Sigmoid)**
```
W ~ Uniform(-√(6/(n_in + n_out)), +√(6/(n_in + n_out)))

or

W ~ Normal(0, √(2/(n_in + n_out)))

Rationale: Preserve variance of activations and gradients
```

**2. He initialization (ReLU)**
```
W ~ Normal(0, √(2/n_in))

Accounts for ReLU killing half the neurons (negative values → 0)
```

**3. LeCun initialization**
```
W ~ Normal(0, √(1/n_in))

For SELU activation
```

**For transformers:**

**Most common (GPT, BERT):**
```
All linear layers: Normal(0, 0.02)

Embeddings: Normal(0, 1/√d_model)

Biases: Zero

Output projection: Scaled by 1/√(2*num_layers) for stability
```

**Why 0.02?**
- Empirical sweet spot
- Small enough for stability
- Large enough to avoid vanishing gradients
- Less principled than He/Xavier but works well

**Layer-wise scaling:**
```
At depth L, scale weights by 1/√L

Prevents gradient explosion in very deep networks
Used in GPT-2/3
```

**Variance preservation principle:**
```
Want: Var(output) ≈ Var(input)
      Var(∂Loss/∂input) ≈ Var(∂Loss/∂output)

For linear layer y = Wx:
  If Var(W) = 1/n_in → Var(y) ≈ Var(x)  ✓
```

**Position embeddings:**

- **Learned**: Initialize like word embeddings
- **Sinusoidal**: Fixed (no initialization)
- **RoPE**: No parameters (rotation matrices)

**Layer norm:**
```
γ (scale): Initialize to 1
β (bias): Initialize to 0

Starts as identity function
```

**Why initialization matters:**

1. **Training speed**: Good init → faster convergence
2. **Final performance**: Bad init can get stuck in poor local minima
3. **Gradient flow**: Prevents vanishing/exploding gradients early on

**Modern practice (transformers):**
- Use library defaults (PyTorch, HuggingFace)
- Small normal (σ=0.02) works well
- Less critical with techniques like layer norm, residual connections

## Follow-up Questions
- Why does ReLU need different initialization than Tanh?
- What happens if you initialize weights too large?
- How does batch normalization affect initialization requirements?
