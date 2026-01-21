# Design an LLM-Based Binary Classifier
# ====================================================================
#
# Design a binary text classifier **without fine-tuning** using only a helper function that scores per-token probabilities under different prompts.
# 
# Given:
# - A function `score_tokens(prompt, text)` that returns log-probabilities for each token in `text` given a `prompt`
# - Two classes: A and B
# 
# Your implementation should:
# 2. **Scoring Mechanism:** Query both prompts and produce a continuous probability
#    estimate P(class=A|x) in [0,1], not just binary labels
# 3. **Numerical Stability:** Handle log-probabilities robustly using log-sum-exp to avoid underflow
# 4. **Aggregation Strategy:** Combine token-level scores into sequence-level scores 
#    (with optional length normalization)
# 5. **Calibration:** Apply Platt scaling or threshold tuning on validation data
# 6. **Evaluation:** Compute ROC-AUC, PR-AUC, precision/recall, F1, and calibration metrics
# 
# **Function signature:**
#
# ====================================================================
import math
import numpy as np
import scipy

def score_tokens(prompt: str, text: str) -> List[float]:
    """
    Helper function: returns log-probabilities for each token in text.
    This is provided by the interviewer (e.g., via LLM API).
    """
    pass
prompt_a = "You are reading positive reviews from our clients: 1)"
prompt_b = "You are reading negative reviews from our clients: 1)"    



def llm_binary_classifier(
    text: str,
    prompt_a: str,
    prompt_b: str,
    score_tokens_fn,
    length_normalize: bool = True
) -> float:
    """
    Classify text using log-likelihood ratio between two prompts.

    Args:
        text: input text to classify
        prompt_a: system prompt for class A
        prompt_b: system prompt for class B
        score_tokens_fn: function(prompt, text) -> List[log_probs]
        length_normalize: whether to normalize by text length

    Returns:
        probability: P(class=A|text) in [0, 1]
    """
    log_probs_a = score_tokens_fn(prompt_a, text)
    log_prob_a = sum(log_probs_a)
    log_probs_b = score_tokens_fn(prompt_b, text)
    log_prob_b = sum(log_probs_b)
    delta = log_prob_a - log_prob_b # if this is positive, log_prob_a > log_prob_b so A is more likely
    if length_normalize:
        delta = delta/len(text)
    # convert log prob back to prob
    return math.exp(delta)    

def platt_scaling_calibration(
    scores: np.ndarray,
    labels: np.ndarray
) -> Tuple[float, float]:
    """
    Learn temperature and bias for Platt scaling calibration.

    Args:
        scores: uncalibrated scores from classifier
        labels: binary labels (0 or 1)

    Returns:
        (temperature, bias): calibration parameters
    """
    # P = sigmoid(s/T + b) = 1/(1 + e^-[s/T + b])
    # so we need to learn T and b 
    # we do this by minimizing cross entropy loss w.r.t. T and b
    from scipy.optimize import minimize
    from scipy.special import expit 
    def cross_entropy_loss(params: List[float]):
        w,b = params # note w = 1/T
        # logistic function = sigmoid function, maps real numbers to probabilities 
        probs = expit(scores * w + b) # Maps score -> linear function of score -> back to probability 
        probs = np.clip(probs, 1e-10, 1-1e-10)
        loss = np.mean((1-labels)*np.log(1-probs) + labels * np.log(probs)) #log maps [0,1] to [-infty, 0] # if probs is high, and label=1, loss is low.
        return loss
    result = minimize(cross_entropy_loss, x0=[0.0, 1.0], method='L-BFGS-B')
    w,b = result.x
    temperature = 1.0/w
    return temperature, b    

