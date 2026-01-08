# Implement Learning Rate Warmup Schedule

**Category:** coding
**Difficulty:** 2
**Tags:** coding, optimization, training, schedules

## Question
Implement learning rate warmup with cosine decay (as used in training transformers).

Your implementation should:
- Linear warmup from 0 to max_lr
- Cosine decay from max_lr to min_lr
- Support setting warmup steps and total steps

**Function signature:**
```python
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=0):
    """
    Compute learning rate with warmup and cosine decay.

    Args:
        step: current training step (0-indexed)
        warmup_steps: number of warmup steps
        max_steps: total number of training steps
        max_lr: maximum learning rate (after warmup)
        min_lr: minimum learning rate (at end of training)
    Returns:
        learning rate for current step
    """
    pass
```

## Answer

**Key concepts:**
1. Warmup phase: lr increases linearly from 0 to max_lr
2. Decay phase: lr decreases following cosine curve
3. Formula: lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))

**Reference implementation:**
```python
import math

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=0):
    """
    Learning rate schedule with linear warmup and cosine decay.
    """
    # Warmup phase: linear increase
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Decay phase: cosine decay
    if step >= max_steps:
        return min_lr

    # Progress through decay phase (0 to 1)
    decay_progress = (step - warmup_steps) / (max_steps - warmup_steps)

    # Cosine decay
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))

    # Scale between min_lr and max_lr
    lr = min_lr + (max_lr - min_lr) * cosine_decay

    return lr

# As a PyTorch scheduler
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, max_steps, max_lr, min_lr=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        lr = get_lr(
            self.current_step,
            self.warmup_steps,
            self.max_steps,
            self.max_lr,
            self.min_lr
        )

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_step += 1
        return lr

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
```

**Alternative schedules:**
```python
def get_lr_linear_decay(step, warmup_steps, max_steps, max_lr, min_lr=0):
    """Linear warmup + linear decay (simpler)."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    if step >= max_steps:
        return min_lr

    decay_progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return max_lr - (max_lr - min_lr) * decay_progress

def get_lr_constant_with_warmup(step, warmup_steps, max_lr):
    """Warmup then constant (no decay)."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    return max_lr

def get_lr_inverse_sqrt(step, warmup_steps, max_lr):
    """
    Warmup + inverse square root decay.
    Used in original Transformer paper.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # After warmup: lr ∝ 1/sqrt(step)
    return max_lr * math.sqrt(warmup_steps / step)
```

**Testing:**
```python
# Test
warmup_steps = 1000
max_steps = 10000
max_lr = 1e-3
min_lr = 1e-5

# Check key points
print("Key learning rate values:")
print(f"Step 0: {get_lr(0, warmup_steps, max_steps, max_lr, min_lr):.2e}")
print(f"Step {warmup_steps//2}: {get_lr(warmup_steps//2, warmup_steps, max_steps, max_lr, min_lr):.2e}")
print(f"Step {warmup_steps}: {get_lr(warmup_steps, warmup_steps, max_steps, max_lr, min_lr):.2e}")
print(f"Step {max_steps//2}: {get_lr(max_steps//2, warmup_steps, max_steps, max_lr, min_lr):.2e}")
print(f"Step {max_steps}: {get_lr(max_steps, warmup_steps, max_steps, max_lr, min_lr):.2e}")

# Visualize schedule
import matplotlib.pyplot as plt

steps = range(0, max_steps + 1, 100)
lrs_cosine = [get_lr(s, warmup_steps, max_steps, max_lr, min_lr) for s in steps]
lrs_linear = [get_lr_linear_decay(s, warmup_steps, max_steps, max_lr, min_lr) for s in steps]
lrs_constant = [get_lr_constant_with_warmup(s, warmup_steps, max_lr) for s in steps]

plt.figure(figsize=(10, 6))
plt.plot(steps, lrs_cosine, label='Cosine Decay')
plt.plot(steps, lrs_linear, label='Linear Decay')
plt.plot(steps, lrs_constant, label='Constant')
plt.axvline(warmup_steps, color='r', linestyle='--', alpha=0.5, label='End of Warmup')
plt.xlabel('Training Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedules')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.show()

# Test with PyTorch optimizer
model = nn.Linear(10, 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

scheduler = WarmupCosineScheduler(
    optimizer,
    warmup_steps=1000,
    max_steps=10000,
    max_lr=1e-3,
    min_lr=1e-5
)

# Simulate training
for step in range(100):
    lr = scheduler.step()
    # ... training code ...

print(f"\nCurrent LR after 100 steps: {scheduler.get_last_lr()[0]:.2e}")
```

**Common mistakes:**
1. ❌ Starting warmup from max_lr instead of 0
2. ❌ Wrong cosine formula (forgetting 0.5 factor)
3. ❌ Not handling step >= max_steps case
4. ❌ Dividing by zero when warmup_steps = 0

## Follow-up Questions
- Why use warmup in transformer training?
- Why cosine decay instead of exponential?
- How do you choose warmup_steps and max_steps?
