import numpy as np

# Implement Loss Functions and Probability Transformations
# ====================================================================
#
# Implement common loss functions and probability transformations from scratch, with proper numerical stability.
# 
# Your implementation should include:
# 
# 1. **Transformations:**
#    - `logits_to_probs`: Convert logits to probabilities using softmax
#    - `probs_to_logprobs`: Convert probabilities to log-probabilities
#    - `logits_to_logprobs`: Convert logits directly to log-probabilities (log-softmax)
# 
# 2. **Loss Functions:**
#    - `cross_entropy_loss`: Categorical cross-entropy from probabilities
#    - `cross_entropy_from_logits`: Cross-entropy directly from logits (more stable)
#    - `binary_cross_entropy`: Binary cross-entropy loss
#    - `kl_divergence`: KL divergence between two distributions
# 
# 3. **Numerical Stability:**
#    - Handle numerical underflow/overflow correctly
#    - Use log-sum-exp trick where appropriate
#    - Avoid computing `log(0)` or `exp(large_number)`
# 
# **Function signatures:**
#
# ====================================================================

def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities using softmax.

    Args:
        logits: shape (batch_size, num_classes) or (num_classes,)
    Returns:
        probs: same shape as logits, values in [0, 1], sum to 1
    """    
    c = np.max(logits, axis=-1, keepdims=True) # handle both logit shapes
    logits = logits - c
    sum_exp = np.sum(np.exp(logits), axis=-1, keepdims=True) # (batch_size, 1) or (1,1)
    return (np.exp(logits) / sum_exp)

def logits_to_logprobs(logits: np.ndarray) -> np.ndarray:
    """Convert logits to log-probabilities using log-softmax.

    More numerically stable than log(softmax(logits)).

    Args:
        logits: shape (batch_size, num_classes) or (num_classes,)
    Returns:
        log_probs: same shape as logits
    """
    # naive way: use log(softmax(x)), but if softmax produces a number near 0
    # log(exp(x)/sum(exp(x))) = log(exp(x)) - log(sum(exp(x)))
    # = x - log(sum(exp(x))) = x - log(sum(exp(x-c)exp(c)))
    # = x - log(exp(c)sum(exp(x-c))) = x - c - log(sum(exp(x-c)))
    c = np.max(logits, axis=-1, keepdims=True) # (batch_size, 1) or (1,)
    return logits - c - np.log(np.sum(np.exp(logits-c), axis=-1, keepdims=True)) 
    
def probs_to_logprobs(probs: np.ndarray) -> np.ndarray:                                    
    """Convert probabilities to log-probabilities.                                         
                                                                                            
    Args:                                                                                  
        probs: shape (batch_size, num_classes) or (num_classes,), values in [0, 1]
    Returns:
        log_probs: same shape as probs
    """
    return np.log(np.clip(probs, 1e-8, None))

def cross_entropy_loss(probs: np.ndarray, targets: np.ndarray) -> float:
    """Compute categorical cross-entropy from probabilities.

    Args:
        probs: predicted probabilities, shape (batch_size, num_classes)
        targets: true class indices, shape (batch_size,)
    Returns:
        loss: scalar cross-entropy loss
    """
    # cross entropy loss is for classification, it's sum_i 1_i * log(p_i)
    # so for the positive class, if you have a prediction near 1 it's almost 0 (slightly neg)
    # if you have a prediction near 0 it approaches -inf
    
    log_preds = probs[np.arange(len(probs)), targets] # use indexing to only select the column of the class
    log_preds = np.log(np.clip(log_preds, 1e-8, None)) # (batch_size, )
    return -np.mean(log_preds)

def cross_entropy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    """Compute cross-entropy directly from logits (numerically stable).

    Args:
        logits: raw model outputs, shape (batch_size, num_classes)
        targets: true class indices, shape (batch_size,)
    Returns:
        loss: scalar cross-entropy loss
    """
    logprobs = logits_to_logprobs(logits)
    nlls = -logprobs[np.arange(len(targets)),targets]
    return np.mean(nlls)

def binary_cross_entropy(probs: np.ndarray, targets: np.ndarray) -> float:
    """Compute binary cross-entropy loss.

    Args:
        probs: predicted probabilities, shape (batch_size,), values in [0, 1]
        targets: binary labels, shape (batch_size,), values in {0, 1}
    Returns:
        loss: scalar BCE loss
    """
    logprobs_1 = np.log(np.clip(probs, 1e-8, None))
    logprobs_0 = np.log(np.clip(1-probs, 1e-8, None))
    lls = targets * logprobs_1 + (1-targets)*logprobs_0
    return -np.mean(lls)

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence KL(p || q) = sum(p * log(p/q)).

    Args:
        p: true distribution, shape (num_classes,)
        q: approximate distribution, shape (num_classes,)
    Returns:
        kl: scalar KL divergence (always >= 0)
    """
    # clip to prevent division by 0, log(0)
    p = np.clip(p, 1e-8, None) 
    q = np.clip(q, 1e-8, None)
    return np.sum(p * np.log(p/q))

def kl_divergence_from_logits(logits_p: np.ndarray, logits_q: np.ndarray) -> float:        
    """Compute KL divergence from logits (numerically stable).                             
                                                                                            
    Args:                                                                                  
        logits_p: logits for true distribution, shape (num_classes,)
        logits_q: logits for approximate distribution, shape (num_classes,)
    Returns:
        kl: scalar KL divergence (always >= 0)
    """
    c_p = np.max(logits_p)
    # compute p = softmax(logits_p - c_p) = softmax(logits_p)
    temp = np.exp(logits_p - c_p)
    p = temp/np.sum(temp, keepdims=True)
    log_p = logits_to_logprobs(p)
    log_q = logits_to_logprobs(q)
    return np.sum(p * (log_p - log_q))