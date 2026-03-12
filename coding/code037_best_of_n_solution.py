# Implement Best-of-N Sampling and Rejection Sampling
# ====================================================================
#
# Implement best-of-N (BoN) sampling and classifier-free guidance — two inference-time safety techniques used at frontier labs.
# 
# **Best-of-N**: Generate N candidate responses, score each with a reward model, and return the best one. This is a simple but effective way to improve quality and safety without retraining.
# 
# **Classifier-free guidance (CFG)**: Blend the logits from a safety-tuned model and a base model to steer generation toward safer outputs.
# 
# Your implementation should include:
# 1. **`best_of_n`**: Generate N candidates and return the highest-scoring one
# 2. **`rejection_sample`**: Keep generating until a candidate passes a safety threshold
# 3. **`classifier_free_guidance`**: Apply CFG to blend base and tuned logits
# 
# **Function signature:**
#
# ====================================================================

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
    candidates = generate_fn(n)
    scores = score_fn(candidates) # tensor of size (batch_size,)
    max_score_ind = torch.argmax(scores)
    return (candidates[max_score_ind], scores[max_score_ind].item())    

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
    for i in range(max_attempts):
        candidate = generate_fn()
        score = score_fn(candidate)
        if score >= threshold:
            return (candidate, score, i+1)
    return (None, 0.0, max_attempts)    

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
    return base_logits + guidance_scale * (tuned_logits-base_logits)

