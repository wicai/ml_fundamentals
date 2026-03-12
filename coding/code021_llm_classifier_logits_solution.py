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
# 1. **Prompt Construction:** Create distinct system prompts for classes A and B that encode discriminative characteristics
# 2. **Scoring Mechanism:** Query both prompts and produce a continuous probability estimate P(class=A|x) in [0,1], not just binary labels
# 3. **Numerical Stability:** Handle log-probabilities robustly using log-sum-exp to avoid underflow
# 4. **Aggregation Strategy:** Combine token-level scores into sequence-level scores (with optional length normalization)
# 5. **Calibration:** Apply Platt scaling or threshold tuning on validation data
# 6. **Evaluation:** Compute ROC-AUC, PR-AUC, precision/recall, F1, and calibration metrics
# 
# **Function signature:**
#
# ====================================================================

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
        llr: raw log-likelihood ratio (not a probability)
    """
    import math
    log_probs_a_per_token = score_tokens_fn(prompt_a, text)
    log_probs_b_per_token = score_tokens_fn(prompt_b, text)
    log_prob_a = sum(log_probs_a_per_token) / (len(log_probs_a_per_token) if length_normalize else 1)
    log_prob_b = sum(log_probs_b_per_token) / (len(log_probs_b_per_token) if length_normalize else 1)
    diff = log_prob_a - log_prob_b
    return diff 
    # this is log(p(x|a)), log(p(x|b))
    # We want P(A|x) = P(x|A)p(A)/p(x)
    # P(B|x) = P(x|B)p(B)/p(x) = 1 - P(A|x)
    # assuming p(A)/p(B) = 1, 
    # p(A|x)/(1-p(A|x)) = p(x|a)/p(x|b)
    # log(p(A|x)/(1-p(A|x))) = log(p(x|a)/p(x|b)) = log_prob_a - log_prob_b = diff
    # let p = p(A|x)
    # log(p/(1-p)) = diff 
    # p/(1-p) = exp(diff)
    # p * (1 + exp(diff)) = exp(diff)
    # p = exp(diff)/(1 + exp(diff))
    # let's think about numeric stability: 
    # if diff is high magnitude and > 0, exp(diff) overflows
    # so we can instead compute 1/(exp(-diff) + 1)
    # but now if diff is high magnitude and < 0, exp(-diff) goes to infinity
    # so we set up a branch

    # if diff > 0:
    #     return 1/(1+math.exp(-diff))
    # else:
    #     return math.exp(diff)/(1 + math.exp(diff))
    

def platt_scaling_calibration(
    scores: np.ndarray,
    labels: np.ndarray
) -> Tuple[float, float]:
    """
    Learn temperature and bias for Platt scaling calibration.

    Args:
        scores: raw log-likelihood ratio scores from classifier
        labels: binary labels (0 or 1)

    Returns:
        (temperature, bias): calibration parameters
    """
    # in platt scaling we have
    # calibrated_prob = sigmoid(temperature * llr + bias)
    # we wanna learn temperature and bias on the scores + labels data
    # so for each score we are gonna compute y_pred = sigmoid(temperature * llr + bias)
    # and compute BCE loss on (y_pred, labels)
    # and backprop that loss to learn temperature and bias 
    from scipy.optimize import minimize
    
    def nll_loss(params):
        temperature, bias = params
        llrs = temperature * scores + bias  # log(p(a)/p(b))
        y_pred = np.clip(1 / ( 1 + np.exp(-llrs)), 1e-15, 1 - 1e-15)
        bce_losses = -(labels * np.log(y_pred) + (1-labels) * np.log(1-y_pred))
        return np.mean(bce_losses)
        
    result = minimize(nll_loss, x0=[1.0, 0.0], method='BFGS')
    return result.x
    
def inference_pipeline(
    val_texts: List[str],
    val_labels: List[int],
    test_texts: List[str],
    prompt_a: str,
    prompt_b: str,
    score_tokens_fn,
) -> List[float]:
    """
    Full pipeline: score validation data, fit Platt scaling, run inference.

    Args:
        val_texts: validation texts for calibration
        val_labels: validation binary labels
        test_texts: texts to classify
        prompt_a: system prompt for class A
        prompt_b: system prompt for class B
        score_tokens_fn: function(prompt, text) -> List[log_probs]

    Returns:
        calibrated_probs: list of P(class=A|text) for each test text
    """
    
    val_uncalibrated_scores = [llm_binary_classifier(text, prompt_a, prompt_b, score_tokens_fn) for text in val_texts]
    temperature, bias = platt_scaling_calibration(val_uncalibrated_scores, val_labels)
    def sigmoid(arr: np.array) -> np.array:
        return 1/(1 + np.exp(-arr))
    test_uncalibrated_scores = [llm_binary_classifier(text, prompt_a, prompt_b, score_tokens_fn) for text in test_texts]
    return [sigmoid(temperature * score + bias) for score in test_uncalibrated_scores]    

