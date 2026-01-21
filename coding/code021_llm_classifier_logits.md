# Design an LLM-Based Binary Classifier

**Category:** coding
**Difficulty:** 3
**Tags:** coding, LLM, classification, logits, probabilities, log-likelihood
**Source:** Anthropic Software Engineer Technical Screen (Sep 2025)

## Question
Design a binary text classifier **without fine-tuning** using only a helper function that scores per-token probabilities under different prompts.

Given:
- A function `score_tokens(prompt, text)` that returns log-probabilities for each token in `text` given a `prompt`
- Two classes: A and B

Your implementation should:
1. **Prompt Construction:** Create distinct system prompts for classes A and B that encode discriminative characteristics
2. **Scoring Mechanism:** Query both prompts and produce a continuous probability estimate P(class=A|x) in [0,1], not just binary labels
3. **Numerical Stability:** Handle log-probabilities robustly using log-sum-exp to avoid underflow
4. **Aggregation Strategy:** Combine token-level scores into sequence-level scores (with optional length normalization)
5. **Calibration:** Apply Platt scaling or threshold tuning on validation data
6. **Evaluation:** Compute ROC-AUC, PR-AUC, precision/recall, F1, and calibration metrics

**Function signature:**
```python
def score_tokens(prompt: str, text: str) -> List[float]:
    """
    Helper function: returns log-probabilities for each token in text.
    This is provided by the interviewer (e.g., via LLM API).
    """
    pass

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
    pass

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
    pass
```

## Answer

**Key Concept:**
Use log-likelihood ratio (LLR) between two prompts to classify text:
- LLR = log P(text | prompt_A) - log P(text | prompt_B)
- Convert to probability: P(A|text) = sigmoid(LLR)
- Apply Platt scaling for calibration

**Reference implementation:**
```python
import numpy as np
from typing import Tuple, List, Callable
from scipy.special import logsumexp
from scipy.optimize import minimize

def llm_binary_classifier(
    text: str,
    prompt_a: str,
    prompt_b: str,
    score_tokens_fn,
    length_normalize: bool = True,
    prior_a: float = 0.5
) -> float:
    """
    Classify text using log-likelihood ratio between two prompts.

    Key insight: Score the same text under two different prompts and
    compare which prompt makes the text more likely.
    """
    # Get log probabilities for text under each prompt
    log_probs_a = score_tokens_fn(prompt_a, text)
    log_probs_b = score_tokens_fn(prompt_b, text)

    # Compute sequence-level log-likelihoods
    # Using log-sum is more stable than summing then taking log
    log_likelihood_a = sum(log_probs_a)
    log_likelihood_b = sum(log_probs_b)

    # Optional: length normalization (divide by number of tokens)
    # Helps prevent bias toward shorter sequences
    if length_normalize:
        log_likelihood_a /= len(log_probs_a)
        log_likelihood_b /= len(log_probs_b)

    # Compute log-likelihood ratio
    # Add prior: log(P(A)/P(B))
    log_prior_ratio = np.log(prior_a / (1 - prior_a))
    llr = log_likelihood_a - log_likelihood_b + log_prior_ratio

    # Convert to probability using sigmoid
    # P(A|text) = 1 / (1 + exp(-LLR))
    prob_a = 1 / (1 + np.exp(-llr))

    return prob_a


def log_sum_exp_stable(log_probs: List[float]) -> float:
    """
    Compute log(sum(exp(log_probs))) in a numerically stable way.

    Prevents underflow when probabilities are very small.
    """
    if not log_probs:
        return -np.inf

    max_log_prob = max(log_probs)
    if np.isinf(max_log_prob):
        return max_log_prob

    # Subtract max for stability
    sum_exp = sum(np.exp(lp - max_log_prob) for lp in log_probs)
    return max_log_prob + np.log(sum_exp)


def platt_scaling_calibration(
    scores: np.ndarray,
    labels: np.ndarray
) -> Tuple[float, float]:
    """
    Learn temperature (T) and bias (b) for Platt scaling.

    Calibrated probability = sigmoid((score - b) / T)

    This corrects for miscalibration in raw scores.
    """
    # Define negative log-likelihood loss
    def nll_loss(params):
        temperature, bias = params
        # Prevent division by zero
        temperature = max(temperature, 1e-10)

        # Calibrated probabilities
        calibrated = 1 / (1 + np.exp(-(scores - bias) / temperature))

        # Negative log-likelihood
        eps = 1e-15  # For numerical stability
        calibrated = np.clip(calibrated, eps, 1 - eps)
        nll = -np.sum(labels * np.log(calibrated) + (1 - labels) * np.log(1 - calibrated))

        return nll

    # Optimize using scipy
    initial_params = [1.0, 0.0]  # Start with T=1, b=0 (identity)
    result = minimize(nll_loss, initial_params, method='BFGS')

    temperature, bias = result.x
    return temperature, bias


def apply_platt_scaling(scores: np.ndarray, temperature: float, bias: float) -> np.ndarray:
    """Apply learned Platt scaling parameters to scores."""
    return 1 / (1 + np.exp(-(scores - bias) / temperature))


# Mock score_tokens function for testing
def mock_score_tokens(prompt: str, text: str) -> List[float]:
    """
    Simulates an LLM API that returns per-token log-probabilities.

    In reality, you'd call an API like:
    - OpenAI: completion with logprobs=True
    - HuggingFace: model.forward() and extract logits
    """
    import random

    # Seed based on prompt+text for consistency
    random.seed(hash(prompt + text) % 10000)

    # Tokenize (simple word-based for demo)
    tokens = text.split()

    # Generate mock log probabilities
    # Sentiment affects the probabilities
    if "positive" in prompt.lower():
        # Positive prompt: higher probs for positive sentiment text
        if any(word in text.lower() for word in ["great", "excellent", "love", "amazing"]):
            base_logprob = -0.5  # High probability
        else:
            base_logprob = -2.0  # Low probability
    else:
        # Negative prompt: higher probs for negative sentiment text
        if any(word in text.lower() for word in ["bad", "terrible", "awful", "hate"]):
            base_logprob = -0.5
        else:
            base_logprob = -2.0

    # Add some randomness
    log_probs = [base_logprob + random.gauss(0, 0.3) for _ in tokens]

    return log_probs


# Build full classifier pipeline
class CalibratedLLMClassifier:
    """
    Full pipeline: prompts + LLR scoring + Platt scaling calibration
    """
    def __init__(self, prompt_a: str, prompt_b: str, score_tokens_fn):
        self.prompt_a = prompt_a
        self.prompt_b = prompt_b
        self.score_tokens_fn = score_tokens_fn
        self.temperature = 1.0
        self.bias = 0.0
        self.calibrated = False

    def predict_proba(self, text: str) -> float:
        """Get probability for class A."""
        score = llm_binary_classifier(
            text,
            self.prompt_a,
            self.prompt_b,
            self.score_tokens_fn
        )

        if self.calibrated:
            # Apply Platt scaling
            score = apply_platt_scaling(
                np.array([score]),
                self.temperature,
                self.bias
            )[0]

        return score

    def calibrate(self, texts: List[str], labels: List[int]):
        """
        Calibrate on validation data using Platt scaling.

        Args:
            texts: validation texts
            labels: binary labels (0 or 1)
        """
        # Get uncalibrated scores
        scores = np.array([
            llm_binary_classifier(
                text,
                self.prompt_a,
                self.prompt_b,
                self.score_tokens_fn
            )
            for text in texts
        ])

        # Learn calibration parameters
        self.temperature, self.bias = platt_scaling_calibration(
            scores,
            np.array(labels)
        )
        self.calibrated = True

        print(f"Calibration: T={self.temperature:.3f}, b={self.bias:.3f}")

    def predict(self, text: str, threshold: float = 0.5) -> int:
        """Predict class label (0 or 1)."""
        prob = self.predict_proba(text)
        return 1 if prob >= threshold else 0
```

**Real-world usage with OpenAI API:**
```python
import openai

def openai_logprobs_api(prompt: str, tokens=["Positive", "Negative"]) -> dict:
    """
    Get log probabilities from OpenAI's API.
    Note: GPT-4 and newer models support logprobs parameter.
    """
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",  # Or other model
        prompt=prompt,
        max_tokens=1,
        temperature=0,
        logprobs=5,  # Return top 5 token probabilities
    )

    # Extract logprobs from response
    top_logprobs = response.choices[0].logprobs.top_logprobs[0]
    return top_logprobs


# Usage
text = "This movie was absolutely fantastic! I loved every minute."
predicted_class, confidence = llm_binary_classifier(
    text,
    openai_logprobs_api,
    class_labels=("Positive", "Negative")
)
print(f"Prediction: {predicted_class} (confidence: {confidence:.2f})")
```

**Testing:**
```python
# Define prompts for sentiment classification
prompt_positive = """You are analyzing text for positive sentiment.
Consider text as coming from satisfied customers who had good experiences.
Rate how likely this text is from that perspective."""

prompt_negative = """You are analyzing text for negative sentiment.
Consider text as coming from dissatisfied customers who had bad experiences.
Rate how likely this text is from that perspective."""

# Create classifier
classifier = CalibratedLLMClassifier(
    prompt_a=prompt_positive,
    prompt_b=prompt_negative,
    score_tokens_fn=mock_score_tokens
)

# Test texts
test_texts = [
    "This product is amazing! Great quality.",
    "Terrible experience, would not recommend.",
    "It's okay, nothing special.",
]

print("=== Before Calibration ===")
for text in test_texts:
    prob = classifier.predict_proba(text)
    pred = "Positive" if prob > 0.5 else "Negative"
    print(f"Text: {text[:50]}")
    print(f"Prediction: {pred} (P(positive)={prob:.3f})\n")


# Calibration data
val_texts = [
    "Excellent service and quality!",
    "Awful quality, broke after one use.",
    "Pretty good, worth the price.",
    "Disappointed with this purchase.",
    "Love this product so much!",
    "Complete waste of money.",
    "Works as expected, no complaints.",
    "Worst purchase ever made.",
]
val_labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

# Calibrate
classifier.calibrate(val_texts, val_labels)

print("\n=== After Calibration ===")
for text in test_texts:
    prob = classifier.predict_proba(text)
    pred = "Positive" if prob > 0.5 else "Negative"
    print(f"Text: {text[:50]}")
    print(f"Prediction: {pred} (P(positive)={prob:.3f})\n")


# Evaluation metrics
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

def evaluate_classifier(classifier, texts, labels):
    """Compute evaluation metrics."""
    probs = [classifier.predict_proba(text) for text in texts]
    preds = [1 if p > 0.5 else 0 for p in probs]

    # ROC-AUC
    auc = roc_auc_score(labels, probs)

    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )

    print(f"ROC-AUC: {auc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")

# Evaluate on validation set
print("\n=== Validation Metrics ===")
evaluate_classifier(classifier, val_texts, val_labels)
```

**Key insights:**

1. **Why use log-likelihood ratio (LLR)?**
   - Scores the same text under two different "worldviews" (prompts)
   - More principled than extracting single token probabilities
   - Handles longer text sequences naturally
   - Mathematically equivalent to Bayes' theorem under certain assumptions

2. **Length normalization is critical:**
   - Without it: longer sequences have lower likelihoods (product of probabilities)
   - Dividing by sequence length prevents bias toward short texts
   - Trade-off: may reduce discriminative power in some cases

3. **Numerical stability matters:**
   - Log probabilities can be very negative (e.g., -100)
   - Use log-sum-exp trick to avoid underflow
   - Never compute exp() then sum, always work in log-space

4. **Calibration significantly improves performance:**
   - Raw LLR scores are often poorly calibrated
   - Platt scaling: learns optimal temperature T and bias b
   - Simple but effective: just two parameters
   - Can improve F1 by 10-20% on typical tasks

5. **Prompt engineering is crucial:**
   - Good prompts encode discriminative characteristics
   - Bad: "This is positive" vs "This is negative"
   - Good: Describe the context/worldview that generates each class
   - Experiment with multiple prompt variations

**Common mistakes:**
1. ❌ Forgetting length normalization (biases toward shorter texts)
2. ❌ Using exp() before summing log-probs (numerical underflow)
3. ❌ Not calibrating on validation data (poor probability estimates)
4. ❌ Using prompts that are too similar (low discrimination)
5. ❌ Ignoring class priors when data is imbalanced

## Follow-up Questions

**Conceptual:**
- Why is log-likelihood ratio mathematically justified? (Hint: Bayes' theorem)
- When would you use LLR vs. fine-tuning a classifier?
- How does Platt scaling relate to logistic regression?
- What are failure modes of this approach? (prompt leakage, length artifacts, etc.)

**Implementation:**
- How would you extend this to multi-class classification (>2 classes)?
- How would you handle very long documents that don't fit in context?
- What if you only have access to the final token's logits, not per-token?
- How would you implement ensemble methods with multiple prompt pairs?

**Practical:**
- How do you validate that prompts are sufficiently discriminative?
- What metrics beyond F1 should you track? (calibration error, ECE, etc.)
- How do you prevent prompt leakage where model memorizes training patterns?
- What's the cost/latency vs. accuracy tradeoff compared to fine-tuning?

**System Design:**
- How would you deploy this in production? (caching, batching, etc.)
- How do you handle model version updates?
- What safety checks are needed before deployment?
- How do you monitor for distribution drift?

## Related Concepts
- Temperature sampling (code008)
- Evaluation metrics (code020)
- Prompt engineering (c047)
- Log-sum-exp numerical stability
- Platt scaling calibration
- ROC-AUC and calibration metrics

## Sources
- [Anthropic Software Engineer Technical Screen Interview Question](https://prachub.com/interview-questions/design-an-llm-based-binary-classifier) (Sep 2025)
- Based on log-likelihood ratio classification theory and Bayesian inference principles
