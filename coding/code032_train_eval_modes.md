# Implement Train vs Eval Mode Effects

**Category:** coding
**Difficulty:** 2
**Tags:** coding, training, pytorch, dropout, batchnorm

## Question
Demonstrate and verify the concrete effects of `model.train()` vs `model.eval()` on model behavior.

Two PyTorch modules behave differently in train vs eval mode:
- **Dropout**: active during training, disabled during eval
- **BatchNorm**: uses batch statistics during training, running statistics during eval

Write functions that demonstrate and verify these differences.

**Function signature:**
```python
def demonstrate_dropout_modes(drop_prob: float = 0.5) -> dict[str, bool]:
    """
    Show that dropout is active in train mode and inactive in eval mode.

    Args:
        drop_prob: dropout probability
    Returns:
        dict with:
            'train_outputs_vary': bool — True if multiple forward passes give
                different results in train mode
            'eval_outputs_same': bool — True if multiple forward passes give
                identical results in eval mode
            'eval_preserves_values': bool — True if eval mode output equals input
                (no dropout applied)
    """
    pass

def demonstrate_batchnorm_modes() -> dict[str, bool]:
    """
    Show that batchnorm uses batch stats in train mode and running stats in eval.

    Returns:
        dict with:
            'train_uses_batch_stats': bool — True if output depends on batch
                composition in train mode
            'eval_uses_running_stats': bool — True if output is the same regardless
                of batch composition in eval mode
            'running_stats_updated_during_train': bool — True if running_mean/var
                change after a forward pass in train mode
    """
    pass

def freeze_for_inference(model: nn.Module) -> nn.Module:
    """
    Prepare a model for inference: eval mode + no gradients.
    Return the model and a context manager for no_grad.

    Args:
        model: nn.Module
    Returns:
        model in eval mode (modified in-place)
    """
    pass
```

## Answer

**Key concepts:**
1. `model.train()` sets `self.training = True` for the module and all submodules
2. `model.eval()` sets `self.training = False` for the module and all submodules
3. Dropout checks `self.training` — drops elements only when True
4. BatchNorm checks `self.training`:
   - True: normalize using current batch mean/var, update running stats
   - False: normalize using stored running_mean/running_var
5. `torch.no_grad()` is separate from eval mode — it controls gradient tracking, not module behavior

**Reference implementation:**
```python
import torch
import torch.nn as nn

def demonstrate_dropout_modes(drop_prob: float = 0.5) -> dict[str, bool]:
    torch.manual_seed(1)
    dropout = nn.Dropout(p=drop_prob)
    x = torch.ones(1, 100)  # All ones — easy to see which are dropped

    # Train mode: dropout is active, outputs vary
    dropout.train()
    train_out1 = dropout(x)
    train_out2 = dropout(x)
    train_outputs_vary = not torch.equal(train_out1, train_out2)

    # Verify scaling: kept values are scaled by 1/(1-p)
    # So the expected value is preserved
    expected_scale = 1.0 / (1.0 - drop_prob)
    nonzero_vals = train_out1[train_out1 != 0]
    scaling_correct = torch.allclose(nonzero_vals, torch.full_like(nonzero_vals, expected_scale))

    # Eval mode: dropout is disabled, output == input
    dropout.eval()
    eval_out1 = dropout(x)
    eval_out2 = dropout(x)
    eval_outputs_same = torch.equal(eval_out1, eval_out2)
    eval_preserves_values = torch.equal(eval_out1, x)

    return {
        'train_outputs_vary': train_outputs_vary,
        'eval_outputs_same': eval_outputs_same,
        'eval_preserves_values': eval_preserves_values,
        'scaling_correct': scaling_correct,
    }

def demonstrate_batchnorm_modes() -> dict[str, bool]:
    torch.manual_seed(1)
    bn = nn.BatchNorm1d(4)

    # Two batches with very different statistics
    batch_a = torch.randn(8, 4) * 1.0  # std ~1
    batch_b = torch.randn(8, 4) * 10.0 + 5.0  # std ~10, mean ~5

    # Train mode: uses batch statistics
    bn.train()
    running_mean_before = bn.running_mean.clone()

    out_a_train = bn(batch_a)
    out_b_train = bn(batch_b)

    # In train mode, output depends on batch composition
    # Both should have roughly mean=0, std=1 (normalized by their own stats)
    train_uses_batch_stats = not torch.allclose(out_a_train, out_b_train[:8], atol=0.1)

    # Running stats should have been updated
    running_stats_updated = not torch.equal(bn.running_mean, running_mean_before)

    # Eval mode: uses running statistics (accumulated during training)
    bn.eval()

    # Process the same input twice — should get same result
    eval_out1 = bn(batch_a)
    eval_out2 = bn(batch_a)
    eval_deterministic = torch.equal(eval_out1, eval_out2)

    # Running stats should NOT change during eval
    running_mean_eval_before = bn.running_mean.clone()
    _ = bn(batch_b)
    running_stats_unchanged_eval = torch.equal(bn.running_mean, running_mean_eval_before)

    return {
        'train_uses_batch_stats': train_uses_batch_stats,
        'eval_uses_running_stats': eval_deterministic,
        'running_stats_updated_during_train': running_stats_updated,
        'running_stats_unchanged_during_eval': running_stats_unchanged_eval,
    }

def freeze_for_inference(model: nn.Module) -> nn.Module:
    """
    Prepare model for inference.
    - eval mode: disables dropout, uses running stats for batchnorm
    - requires_grad_(False): disables gradient computation for parameters
    """
    model.eval()
    model.requires_grad_(False)
    return model
```

**Testing:**
```python
import torch
import torch.nn as nn

# Test 1: Dropout modes
print("=" * 70)
print("TEST 1: Dropout Train vs Eval")
print("=" * 70)
results = demonstrate_dropout_modes()
for key, value in results.items():
    status = "PASS" if value else "FAIL"
    print(f"  {key}: {value} [{status}]")

# Test 2: BatchNorm modes
print("\n" + "=" * 70)
print("TEST 2: BatchNorm Train vs Eval")
print("=" * 70)
results = demonstrate_batchnorm_modes()
for key, value in results.items():
    status = "PASS" if value else "FAIL"
    print(f"  {key}: {value} [{status}]")

# Test 3: freeze_for_inference
print("\n" + "=" * 70)
print("TEST 3: Freeze for Inference")
print("=" * 70)
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(32, 5),
)
model = freeze_for_inference(model)
print(f"model.training = {model.training} (should be False)")
all_frozen = all(not p.requires_grad for p in model.parameters())
print(f"All params frozen: {all_frozen} (should be True)")

# Forward pass should work without building grad graph
x = torch.randn(4, 10)
out = model(x)
print(f"Output requires_grad: {out.requires_grad} (should be False)")
print(f"Output shape: {out.shape}")

# Test 4: Common bug — forgetting to switch modes
print("\n" + "=" * 70)
print("TEST 4: Impact of Forgetting model.eval()")
print("=" * 70)
torch.manual_seed(1)
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.Dropout(0.5),
    nn.Linear(32, 2),
)
x = torch.randn(100, 10)

# With eval: deterministic
model.eval()
out1 = model(x)
out2 = model(x)
print(f"Eval mode — outputs identical: {torch.equal(out1, out2)}")

# Without eval (train mode): stochastic!
model.train()
out3 = model(x)
out4 = model(x)
print(f"Train mode — outputs identical: {torch.equal(out3, out4)}")
print("Forgetting model.eval() gives different results each forward pass!")
```

**Common mistakes:**
1. Forgetting `model.eval()` before validation/inference (dropout still active!)
2. Forgetting `model.train()` when resuming training after evaluation
3. Confusing `model.eval()` with `torch.no_grad()` — they do different things
4. Not realizing BatchNorm running stats are updated during training forward passes
5. Thinking `model.eval()` stops gradient computation (it doesn't — use `torch.no_grad()` for that)

## Follow-up Questions
- What's the difference between `model.eval()` and `torch.no_grad()`?
- Why does dropout scale values by `1/(1-p)` during training?
- What happens if you train with a batch size of 1 with BatchNorm?
- How does `model.requires_grad_(False)` differ from `torch.no_grad()`?
- What other modules might behave differently in train vs eval mode?
