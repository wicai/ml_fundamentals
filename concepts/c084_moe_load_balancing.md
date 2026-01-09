# MoE Load Balancing

**Category:** modern_llm
**Difficulty:** 4
**Tags:** architecture, moe, training

## Question
What is the load balancing problem in Mixture of Experts and how do we solve it?

## What to Cover
- **Set context by**: Explaining why load imbalance happens (self-reinforcing feedback loop where popular experts get more training)
- **Must mention**: Solutions (auxiliary load balancing loss, capacity factor and dropping, expert choice routing), how to tune (α≈0.01), metrics to track
- **Show depth by**: Giving examples from real systems (Switch Transformer, Mixtral) and discussing tradeoffs (too much balancing hurts routing quality)
- **Avoid**: Only describing the problem without explaining the solutions and how to tune them

## Answer
**Load balancing problem**: In MoE models, some experts get overused while others are underused, wasting capacity.

**Why it happens:**

```
MoE with 8 experts, top-1 routing:

Ideal (balanced):
  Expert 1: 12.5% of tokens
  Expert 2: 12.5% of tokens
  ...
  Expert 8: 12.5% of tokens

Reality (collapsed):
  Expert 1: 80% of tokens  (overloaded!)
  Expert 2: 15% of tokens
  Expert 3-8: <1% of tokens each  (wasted capacity!)
```

**Root cause: Training dynamics**

```
1. Early training: Expert 1 slightly better by chance
2. Router learns to send more tokens to Expert 1
3. Expert 1 gets more gradients → learns faster
4. Router sends even more tokens to Expert 1
5. Repeat → Expert 1 dominates

Self-reinforcing feedback loop!
```

**Why it's bad:**

1. **Wasted parameters**: Unused experts contribute nothing
2. **Reduced capacity**: Model effectively smaller than total params
3. **Training instability**: Gradient variance increases
4. **Inference efficiency**: Can't parallelize well if one expert is bottleneck

**Solution 1: Auxiliary load balancing loss**

```
Main loss: L_task (language modeling)
Auxiliary loss: L_balance (encourage uniform expert usage)

Total loss = L_task + α * L_balance

where α is small (0.01 - 0.1)
```

**How the auxiliary loss works:**

```python
# For a batch of tokens
expert_counts = [num_tokens_routed_to_expert_i for i in range(num_experts)]
ideal_count = total_tokens / num_experts

# L1 load balancing (used in Switch Transformer)
L_balance = Σ |expert_counts[i] - ideal_count|

# L2 load balancing (used in GShard)
L_balance = Σ (expert_counts[i] - ideal_count)²

# Importance loss (used in many systems)
# Encourages equal importance scores
importance = Σ router_scores[i]  # Sum scores for each expert
L_balance = CV(importance)²  # Coefficient of variation
```

**Solution 2: Capacity factor and dropping**

```
Capacity per expert = (tokens_per_batch / num_experts) × capacity_factor

Example with capacity_factor = 1.25:
  Batch: 1000 tokens
  Experts: 8
  Base capacity: 1000/8 = 125 tokens per expert
  Actual capacity: 125 × 1.25 = 156 tokens per expert

If >156 tokens route to an expert:
  Keep top 156 by router score
  Drop the rest (skip that expert for those tokens)
```

**Dropping forces diversity:**
- Popular expert reaches capacity → tokens forced to other experts
- Other experts get training signal
- Prevents total collapse

**Solution 3: Expert choice routing**

Standard: **Tokens choose experts**
```
Token: "Which expert should I use?"
Router: "Expert 3 and Expert 7"

Problem: All tokens might choose Expert 3!
```

Expert choice: **Experts choose tokens**
```
Each expert: "I'll process these k tokens"
Each expert gets exactly k tokens

Automatically balanced by design!
```

**Solution 4: Random routing with load balancing**

```
# Instead of pure top-k selection
scores = softmax(router(token))  # [num_experts]

# Sample k experts weighted by scores
experts = sample_without_replacement(scores, k=2)

Adds exploration → prevents collapse
```

**Solution 5: Staged training**

```
Phase 1 (first 10% of training):
  Use dense model (no MoE)
  OR use heavy load balancing (α=1.0)

Phase 2 (remaining 90%):
  Enable MoE
  Reduce α gradually (1.0 → 0.01)

Prevents early collapse
```

**Metrics to track load balancing:**

```python
# 1. Expert usage entropy (higher is better)
usage = [expert_counts[i]/total for i in range(num_experts)]
entropy = -Σ usage[i] * log(usage[i])

Max entropy = log(num_experts)  # Perfect balance

# 2. Load balance factor
expert_fraction = expert_counts / total_tokens
ideal = 1.0 / num_experts
balance_factor = 1 - max(expert_fraction) / ideal

1.0 = perfect, 0.0 = collapsed to one expert

# 3. Expert utilization
utilization = num_active_experts / num_total_experts
```

**Real-world examples:**

**Switch Transformer (Google):**
- Top-1 routing with capacity factor 1.0-2.0
- Auxiliary loss: importance balancing
- Result: 95%+ expert utilization

**GLaM (Google):**
- Top-2 routing with capacity factor 2.0
- Load balancing loss coefficient α=0.01
- Random noise in router scores

**Mixtral 8x7B (Mistral):**
- Top-2 routing
- Small load balancing loss
- Expert choice routing (rumored)

**Tradeoffs:**

```
Strong load balancing (high α):
  ✓ All experts used equally
  ✗ Routing quality suffers (forced to use suboptimal experts)

Weak load balancing (low α):
  ✓ Router picks best experts
  ✗ Risk of collapse

Sweet spot: α ≈ 0.01
```

**Debugging load imbalance:**

```
Symptoms:
- Validation loss stops improving
- Some experts have near-zero gradients
- Router entropy decreases over time

Fixes:
- Increase load balancing coefficient
- Add noise to router
- Reset router weights
- Increase capacity factor
```

## Follow-up Questions
- How does expert parallelism interact with load balancing?
- Can you do MoE without load balancing losses?
- What happens if you set capacity factor to 1.0?
