# Activation Checkpointing / Gradient Checkpointing

**Category:** training
**Difficulty:** 3
**Tags:** training, memory, optimization

## Question
What is activation checkpointing and when should you use it?

## What to Cover
- **Set context by**: Explaining that activations dominate memory during training (stored for backward pass)
- **Must mention**: The tradeoff (recompute activations to save memory), memory vs compute cost, typical savings (2-4× batch size increase)
- **Show depth by**: Mentioning checkpointing strategies (every k layers, selective) and when NOT to use (already compute-bound)
- **Avoid**: Only describing the concept without quantifying the memory-compute tradeoff

## Answer
**Problem**: Backpropagation needs activations from forward pass → stored in memory.

**Memory cost:**
```
Activations: batch_size × seq_len × d_model × num_layers
Example: 32 × 2048 × 4096 × 32 = 8.6GB

This dominates memory for large batch/sequence!
```

**Activation Checkpointing**: Trade compute for memory by recomputing activations.

**Standard backprop:**
```
Forward: Compute and store all activations
Backward: Use stored activations for gradients
Memory: O(num_layers)
```

**With checkpointing:**
```
Forward: Compute activations, store only subset (checkpoints)
Backward:
  1. Recompute activations from last checkpoint
  2. Use recomputed activations for gradients
  3. Discard recomputed activations
Memory: O(sqrt(num_layers)) or O(1) depending on strategy
```

**Strategies:**

**1. Checkpoint every k layers:**
```
32 layers, checkpoint every 4:
  Store: layers [0, 4, 8, 12, 16, 20, 24, 28]
  Recompute: At most 4 layers during backward
```

**2. Selective checkpointing:**
```
Store:
  - Attention outputs (expensive to compute)
  - Layer norm outputs (cheap, store them)

Skip:
  - FFN outputs (expensive, recompute)
```

**Memory-Compute trade-off:**
```
No checkpointing:
  Memory: 100GB activations
  Compute: 1× forward, 1× backward

Full checkpointing (every layer):
  Memory: 10GB activations (10× savings)
  Compute: 1× forward, 2× backward (2× slower)
```

**Typical speedup: 20-30% slower for 50% memory savings.**

**When to use:**

✓ Want larger batch size
✓ Want longer sequences
✓ Limited GPU memory
✗ Already maxing out GPUs (compute-bound)

**Implementation:**
```python
# PyTorch
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    # Checkpoint this expensive block
    x = checkpoint(self.expensive_layer, x)
    return x
```

**Modern LLM training:**
- Almost always used
- Allows 2-4× larger batch size
- Critical for training with long contexts (4K-32K tokens)

## Follow-up Questions
- What's the optimal checkpointing frequency?
- How much does checkpointing slow down training?
- Can you checkpoint within a single transformer layer?
