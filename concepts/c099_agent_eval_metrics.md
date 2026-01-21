# Agent Evaluation Metrics: pass@k vs pass^k

**Category:** agents
**Difficulty:** 3
**Tags:** agents, evaluation, metrics

## Question
What's the difference between pass@k and pass^k metrics for agent evaluation? When should you use each?

## What to Cover
- **Set context by**: Explaining that agents are non-deterministic, so multiple trials are needed to measure true capability
- **Must mention**: Definitions of pass@k (at least one success in k tries) and pass^k (all k tries succeed), formulas, and when to use each
- **Show depth by**: Discussing implications for production (retry-allowed vs single-shot), how to choose k, and the relationship between the metrics
- **Avoid**: Only giving definitions without explaining the practical implications for agent deployment

## Answer
**Why multiple trials?**

Agents are non-deterministic:
- Same task, different outcomes each run
- LLM sampling variability
- Tool/environment variability
- Need multiple trials to measure true capability

**Key terminology:**
- **Task**: A single test with inputs and success criteria
- **Trial**: One attempt at a task
- **Transcript**: Complete record of agent actions in a trial

**pass@k: At least one success**

```
pass@k = P(at least 1 success in k trials)
```

**Formula:**
```
pass@k = 1 - (1 - p)^k

where p = single-trial success rate
```

**Example:**
```
If p = 0.3 (30% per trial):

pass@1 = 0.30 (30%)
pass@3 = 1 - (0.7)^3 = 0.657 (66%)
pass@5 = 1 - (0.7)^5 = 0.832 (83%)
pass@10 = 1 - (0.7)^10 = 0.972 (97%)
```

**Interpretation:**
- "Can the agent ever solve this task?"
- Higher k → higher pass@k (easier to pass)
- Measures capability ceiling

**Use when:**
- Retries are allowed in production
- You only need one correct answer
- Exploring agent capabilities
- Code generation (run tests, retry if fails)

**pass^k: All trials succeed**

```
pass^k = P(all k trials succeed)
```

**Formula:**
```
pass^k = p^k

where p = single-trial success rate
```

**Example:**
```
If p = 0.9 (90% per trial):

pass^1 = 0.90 (90%)
pass^3 = (0.9)^3 = 0.729 (73%)
pass^5 = (0.9)^5 = 0.590 (59%)
pass^10 = (0.9)^10 = 0.349 (35%)
```

**Interpretation:**
- "Is the agent reliably consistent?"
- Higher k → lower pass^k (harder to pass)
- Measures reliability/consistency

**Use when:**
- No retries in production
- Consistency is critical
- Safety-critical applications
- User-facing single-shot interactions

**Comparison:**

```
Single-trial p = 0.7

k    | pass@k | pass^k
-----|--------|-------
1    | 70%    | 70%
3    | 97%    | 34%
5    | 99.8%  | 17%
10   | 99.99% | 3%
```

**Visual intuition:**
```
pass@k: "Did ANY dart hit the target?"
  Trial 1: Miss
  Trial 2: Hit ✓  → pass@k = success
  Trial 3: Miss

pass^k: "Did ALL darts hit the target?"
  Trial 1: Miss  → pass^k = fail
  Trial 2: Hit
  Trial 3: Miss
```

**Choosing the right metric:**

```python
def choose_metric(use_case):
    if use_case == "capability_benchmark":
        # Testing what's possible
        return "pass@k with k=10"

    elif use_case == "production_with_retry":
        # Can retry a few times
        return "pass@k with k=3"

    elif use_case == "production_single_shot":
        # User expects one good response
        return "pass^k with k=5"

    elif use_case == "safety_critical":
        # Must be reliable
        return "pass^k with k=10"
```

**Estimating from trials:**

```python
def estimate_metrics(task_results, k):
    """
    task_results: list of (task_id, [trial1_pass, trial2_pass, ...])
    """
    pass_at_k_scores = []
    pass_pow_k_scores = []

    for task_id, trials in task_results:
        n = len(trials)
        c = sum(trials)  # number of successes

        # Unbiased estimator for pass@k
        # Avoids sampling bias
        if n - c < k:
            pass_at_k = 1.0
        else:
            pass_at_k = 1.0 - comb(n-c, k) / comb(n, k)

        # For pass^k, simple estimate
        p_hat = c / n
        pass_pow_k = p_hat ** k

        pass_at_k_scores.append(pass_at_k)
        pass_pow_k_scores.append(pass_pow_k)

    return {
        "pass@k": np.mean(pass_at_k_scores),
        "pass^k": np.mean(pass_pow_k_scores)
    }
```

**Practical recommendations:**

**1. Report both metrics:**
```
Results on WebArena:
- pass@1: 34% (single-shot capability)
- pass@3: 52% (with retries)
- pass^3: 18% (consistency)
```

**2. Choose k based on production constraints:**
```
# If you can retry 3 times in production
# Report pass@3 as your "capability" metric
# Report pass^3 as your "reliability" metric
```

**3. Track over time:**
```
Week 1: pass@3=60%, pass^3=20%
Week 2: pass@3=65%, pass^3=30%  # Both improving
Week 3: pass@3=70%, pass^3=25%  # Capability up, consistency down!
```

**4. Segment by task difficulty:**
```
Easy tasks:   pass@3=95%, pass^3=85%
Medium tasks: pass@3=70%, pass^3=40%
Hard tasks:   pass@3=30%, pass^3=5%
```

**Common pitfalls:**

1. **Only reporting pass@k with high k**
   - Makes agent look better than production reality
   - Always report pass@1 as baseline

2. **Ignoring pass^k for production systems**
   - Users don't get unlimited retries
   - Consistency matters for trust

3. **Not running enough trials**
   - Need sufficient n to estimate p accurately
   - Rule of thumb: n ≥ 10 trials per task

## Follow-up Questions
- How many trials do you need to get reliable estimates?
- How does pass@k relate to cost in production (more retries = more cost)?
- When would you use a metric like "majority vote of k trials"?
