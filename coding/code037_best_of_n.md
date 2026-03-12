# Implement Best-of-N Sampling and Rejection Sampling

**Category:** coding
**Difficulty:** 3
**Tags:** coding, safety, inference, reward model, alignment

## Question
Implement best-of-N (BoN) sampling and classifier-free guidance — two inference-time safety techniques used at frontier labs.

**Best-of-N**: Generate N candidate responses, score each with a reward model, and return the best one. This is a simple but effective way to improve quality and safety without retraining.

**Classifier-free guidance (CFG)**: Blend the logits from a safety-tuned model and a base model to steer generation toward safer outputs.

Your implementation should include:
1. **`best_of_n`**: Generate N candidates and return the highest-scoring one
2. **`rejection_sample`**: Keep generating until a candidate passes a safety threshold
3. **`classifier_free_guidance`**: Apply CFG to blend base and tuned logits

**Function signature:**
```python
def best_of_n(
    generate_fn: Callable[[int], list[str]],
    score_fn: Callable[[list[str]], torch.Tensor],
    n: int = 16,
) -> tuple[str, float]:
    """
    Best-of-N sampling: generate N candidates and return the best.

    Args:
        generate_fn: function that takes batch_size and returns list of generated strings
        score_fn: function that takes list of strings and returns reward scores (batch_size,)
        n: number of candidates to generate
    Returns:
        best_response: the highest-scoring candidate
        best_score: its reward score
    """
    pass

def rejection_sample(
    generate_fn: Callable[[], str],
    score_fn: Callable[[str], float],
    threshold: float,
    max_attempts: int = 100,
) -> tuple[str | None, float, int]:
    """
    Rejection sampling: keep generating until a candidate exceeds the threshold.

    Args:
        generate_fn: function that returns a single generated string
        score_fn: function that returns a scalar reward for a string
        threshold: minimum acceptable reward score
        max_attempts: give up after this many attempts
    Returns:
        response: the first passing candidate (None if all failed)
        score: its score (0.0 if None)
        attempts: number of attempts made
    """
    pass

def classifier_free_guidance(
    base_logits: torch.Tensor,
    tuned_logits: torch.Tensor,
    guidance_scale: float = 1.5,
) -> torch.Tensor:
    """
    Classifier-free guidance: steer logits toward the tuned model.

    guided_logits = base_logits + guidance_scale * (tuned_logits - base_logits)
                  = (1 - guidance_scale) * base_logits + guidance_scale * tuned_logits

    When guidance_scale = 1.0: just tuned_logits (no guidance)
    When guidance_scale > 1.0: amplify the difference (stronger steering)
    When guidance_scale = 0.0: just base_logits (no tuning)

    Args:
        base_logits: logits from base model, shape (batch, vocab_size)
        tuned_logits: logits from safety-tuned model, shape (batch, vocab_size)
        guidance_scale: how strongly to steer. >1 amplifies safety tuning.
    Returns:
        guided_logits: shape (batch, vocab_size)
    """
    pass
```

## Answer

**Key concepts:**
1. Best-of-N is compute-expensive but simple: quality scales as O(log N) reward improvement
2. Rejection sampling is useful when you need a hard safety threshold
3. CFG amplifies the "direction" the tuned model moved from the base model
4. These are all inference-time techniques — no retraining needed
5. Best-of-N with a reward model is equivalent to importance sampling from the optimal policy

**Reference implementation:**
```python
import torch
from typing import Callable

def best_of_n(
    generate_fn: Callable[[int], list[str]],
    score_fn: Callable[[list[str]], torch.Tensor],
    n: int = 16,
) -> tuple[str, float]:
    """Generate N candidates and return the highest-scoring one."""
    # Generate all candidates
    candidates = generate_fn(n)

    # Score all candidates
    scores = score_fn(candidates)  # (n,)

    # Select the best
    best_idx = scores.argmax().item()
    return candidates[best_idx], scores[best_idx].item()

def rejection_sample(
    generate_fn: Callable[[], str],
    score_fn: Callable[[str], float],
    threshold: float,
    max_attempts: int = 100,
) -> tuple[str | None, float, int]:
    """Keep generating until a candidate exceeds the threshold."""
    for attempt in range(1, max_attempts + 1):
        candidate = generate_fn()
        score = score_fn(candidate)

        if score >= threshold:
            return candidate, score, attempt

    # All attempts failed
    return None, 0.0, max_attempts

def classifier_free_guidance(
    base_logits: torch.Tensor,
    tuned_logits: torch.Tensor,
    guidance_scale: float = 1.5,
) -> torch.Tensor:
    """
    CFG: amplify the difference between tuned and base.

    guided = base + scale * (tuned - base) = (1 - scale) * base + scale * tuned
    """
    return base_logits + guidance_scale * (tuned_logits - base_logits)
```

**Testing:**
```python
import torch
import random

torch.manual_seed(1)
random.seed(1)

# Test 1: Best-of-N basic
print("=" * 70)
print("TEST 1: Best-of-N Sampling")
print("=" * 70)

# Simulate: generate_fn returns random strings, score_fn scores them
responses = [f"Response {i}: {'good' if i % 3 == 0 else 'ok'}" for i in range(20)]

def mock_generate(n: int) -> list[str]:
    return random.sample(responses, n)

def mock_score(texts: list[str]) -> torch.Tensor:
    return torch.tensor([1.0 if 'good' in t else 0.3 for t in texts])

best, score = best_of_n(mock_generate, mock_score, n=8)
print(f"Best response: {best}")
print(f"Best score: {score:.2f}")
print(f"Selected a 'good' response: {'good' in best}")

# Test 2: Best-of-N improves with larger N
print("\n" + "=" * 70)
print("TEST 2: Quality Improves with N")
print("=" * 70)

def noisy_generate(n: int) -> list[str]:
    return [f"resp_{i}" for i in range(n)]

def noisy_score(texts: list[str]) -> torch.Tensor:
    return torch.randn(len(texts))  # Random scores

avg_scores = {}
for n in [1, 4, 16, 64]:
    scores_at_n = []
    for _ in range(100):
        _, score = best_of_n(noisy_generate, noisy_score, n=n)
        scores_at_n.append(score)
    avg_scores[n] = sum(scores_at_n) / len(scores_at_n)
    print(f"N={n:3d}: avg best score = {avg_scores[n]:.3f}")

print(f"Score increases with N: {avg_scores[64] > avg_scores[16] > avg_scores[4] > avg_scores[1]}")

# Test 3: Rejection sampling
print("\n" + "=" * 70)
print("TEST 3: Rejection Sampling")
print("=" * 70)

call_count = 0
def counted_generate() -> str:
    global call_count
    call_count += 1
    return f"response_{call_count}"

def threshold_score(text: str) -> float:
    # Only pass on attempt 5
    num = int(text.split('_')[1])
    return 1.0 if num >= 5 else 0.2

call_count = 0
result, score, attempts = rejection_sample(counted_generate, threshold_score, threshold=0.5)
print(f"Result: {result}")
print(f"Score: {score:.2f}")
print(f"Attempts: {attempts} (should be 5)")

# Test: all fail
call_count = 0
result, score, attempts = rejection_sample(
    counted_generate, lambda x: 0.1, threshold=0.5, max_attempts=10
)
print(f"All fail — result: {result} (should be None)")
print(f"Attempts: {attempts} (should be 10)")

# Test 4: Classifier-free guidance
print("\n" + "=" * 70)
print("TEST 4: Classifier-Free Guidance")
print("=" * 70)

base_logits = torch.tensor([[2.0, 1.0, 0.5, 0.1]])   # Base prefers token 0
tuned_logits = torch.tensor([[1.5, 1.0, 2.0, 0.1]])   # Tuned prefers token 2

for scale in [0.0, 1.0, 1.5, 3.0]:
    guided = classifier_free_guidance(base_logits, tuned_logits, scale)
    probs = torch.softmax(guided, dim=-1)
    preferred = guided.argmax(dim=-1).item()
    print(f"scale={scale:.1f}: logits={guided[0].tolist()}, preferred=token_{preferred}, "
          f"probs={[f'{p:.3f}' for p in probs[0].tolist()]}")

# At scale=0: should be base_logits
guided_0 = classifier_free_guidance(base_logits, tuned_logits, 0.0)
print(f"\nscale=0 matches base: {torch.allclose(guided_0, base_logits)}")

# At scale=1: should be tuned_logits
guided_1 = classifier_free_guidance(base_logits, tuned_logits, 1.0)
print(f"scale=1 matches tuned: {torch.allclose(guided_1, tuned_logits)}")

# At scale>1: amplifies difference
guided_2 = classifier_free_guidance(base_logits, tuned_logits, 2.0)
diff_tuned = (tuned_logits - base_logits)[0, 2].item()
diff_guided = (guided_2 - base_logits)[0, 2].item()
print(f"scale=2 amplifies diff by 2x: {abs(diff_guided / diff_tuned - 2.0) < 1e-5}")
```

**Common mistakes:**
1. Not handling the case where all rejection sampling attempts fail
2. CFG formula sign error: it's `base + scale * (tuned - base)`, not `tuned + scale * (base - tuned)`
3. Using argmax for best-of-N instead of proper scoring (argmax is greedy over vocab, not over candidates)
4. Not understanding that best-of-N quality scales as O(log N), not O(N)

## Follow-up Questions
- How does the expected reward of best-of-N scale with N?
- What is the "alignment tax" and how does best-of-N avoid it?
- How does CFG relate to RLHF? What are the trade-offs?
- When would you use rejection sampling vs best-of-N?
- What is the KL cost of best-of-N sampling? (It's KL = log(N) - (N-1)/N)
