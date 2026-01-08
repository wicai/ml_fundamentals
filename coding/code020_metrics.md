# Implement ML Evaluation Metrics

**Category:** coding
**Difficulty:** 2
**Tags:** coding, evaluation, metrics, classification

## Question
Implement common machine learning evaluation metrics from scratch.

Your implementation should include:
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC (binary classification)
- Top-k Accuracy

**Function signatures:**
```python
def accuracy(y_true, y_pred):
    """Compute accuracy."""
    pass

def precision_recall_f1(y_true, y_pred, average='binary'):
    """Compute precision, recall, and F1."""
    pass

def confusion_matrix(y_true, y_pred, num_classes):
    """Compute confusion matrix."""
    pass

def roc_auc(y_true, y_scores):
    """Compute ROC-AUC score."""
    pass

def top_k_accuracy(y_true, y_pred_probs, k=5):
    """Compute top-k accuracy."""
    pass
```

## Answer

**Reference implementation:**
```python
import torch
import numpy as np

def accuracy(y_true, y_pred):
    """
    Compute classification accuracy.

    Args:
        y_true: ground truth labels (N,)
        y_pred: predicted labels (N,)
    Returns:
        accuracy: float in [0, 1]
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    correct = (y_true == y_pred).sum()
    total = len(y_true)

    return correct / total if total > 0 else 0.0

def confusion_matrix(y_true, y_pred, num_classes):
    """
    Compute confusion matrix.

    Args:
        y_true: ground truth labels (N,)
        y_pred: predicted labels (N,)
        num_classes: number of classes
    Returns:
        confusion matrix (num_classes, num_classes)
        rows = true labels, cols = predicted labels
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1

    return cm

def precision_recall_f1(y_true, y_pred, average='binary', pos_label=1):
    """
    Compute precision, recall, and F1 score.

    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        average: 'binary', 'macro', or 'micro'
        pos_label: positive class label for binary classification
    Returns:
        (precision, recall, f1): tuple of floats
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    if average == 'binary':
        # Binary classification
        tp = ((y_true == pos_label) & (y_pred == pos_label)).sum()
        fp = ((y_true != pos_label) & (y_pred == pos_label)).sum()
        fn = ((y_true == pos_label) & (y_pred != pos_label)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    elif average == 'macro':
        # Macro averaging: compute per-class then average
        classes = np.unique(np.concatenate([y_true, y_pred]))
        precisions, recalls, f1s = [], [], []

        for c in classes:
            tp = ((y_true == c) & (y_pred == c)).sum()
            fp = ((y_true != c) & (y_pred == c)).sum()
            fn = ((y_true == c) & (y_pred != c)).sum()

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        return np.mean(precisions), np.mean(recalls), np.mean(f1s)

    elif average == 'micro':
        # Micro averaging: aggregate all TP, FP, FN
        classes = np.unique(np.concatenate([y_true, y_pred]))
        total_tp = total_fp = total_fn = 0

        for c in classes:
            total_tp += ((y_true == c) & (y_pred == c)).sum()
            total_fp += ((y_true != c) & (y_pred == c)).sum()
            total_fn += ((y_true == c) & (y_pred != c)).sum()

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

def roc_auc(y_true, y_scores):
    """
    Compute ROC-AUC score for binary classification.

    Args:
        y_true: ground truth binary labels (N,)
        y_scores: predicted scores/probabilities (N,)
    Returns:
        auc: area under ROC curve
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()

    # Sort by scores (descending)
    sorted_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[sorted_indices]

    # Count positives and negatives
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()

    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined, return random guess

    # Compute TPR and FPR at each threshold
    tp = 0
    fp = 0
    auc = 0.0

    prev_tp = 0
    prev_fp = 0

    for label in y_true_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
            # When we get a negative, add the area of the trapezoid
            # Area = (tp_current + tp_prev) / 2 * (fp_current - fp_prev) / n_pos / n_neg
            auc += tp

    # Normalize
    auc = auc / (n_pos * n_neg)

    return auc

def top_k_accuracy(y_true, y_pred_probs, k=5):
    """
    Compute top-k accuracy.

    Args:
        y_true: ground truth labels (N,)
        y_pred_probs: predicted probabilities (N, num_classes)
        k: number of top predictions to consider
    Returns:
        top_k_acc: float in [0, 1]
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred_probs, torch.Tensor):
        y_pred_probs = y_pred_probs.cpu().numpy()

    # Get top-k predicted classes
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]

    # Check if true label is in top-k
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1

    return correct / len(y_true)
```

**Testing:**
```python
# Test accuracy
y_true = torch.tensor([0, 1, 2, 0, 1, 2])
y_pred = torch.tensor([0, 2, 2, 0, 1, 1])

acc = accuracy(y_true, y_pred)
print(f"Accuracy: {acc:.4f}")  # 4/6 = 0.6667

# Test confusion matrix
cm = confusion_matrix(y_true, y_pred, num_classes=3)
print(f"\nConfusion Matrix:\n{cm}")

# Test precision, recall, F1
y_true_binary = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0])
y_pred_binary = torch.tensor([1, 0, 1, 0, 0, 1, 0, 1])

p, r, f1 = precision_recall_f1(y_true_binary, y_pred_binary, average='binary')
print(f"\nPrecision: {p:.4f}")
print(f"Recall: {r:.4f}")
print(f"F1: {f1:.4f}")

# Test ROC-AUC
y_scores = torch.tensor([0.9, 0.1, 0.8, 0.4, 0.2, 0.7, 0.3, 0.6])
auc = roc_auc(y_true_binary, y_scores)
print(f"\nROC-AUC: {auc:.4f}")

# Compare with sklearn
from sklearn.metrics import roc_auc_score
sklearn_auc = roc_auc_score(y_true_binary.numpy(), y_scores.numpy())
print(f"Sklearn ROC-AUC: {sklearn_auc:.4f}")

# Test top-k accuracy
y_true_multi = torch.tensor([2, 0, 3, 1, 2])
y_pred_probs = torch.tensor([
    [0.1, 0.2, 0.3, 0.4],  # True: 2, Top-2: [3, 2] ✓
    [0.8, 0.1, 0.05, 0.05], # True: 0, Top-2: [0, 1] ✓
    [0.2, 0.2, 0.2, 0.4],  # True: 3, Top-2: [3, 0/1/2] ✓
    [0.1, 0.6, 0.2, 0.1],  # True: 1, Top-2: [1, 2] ✓
    [0.4, 0.3, 0.2, 0.1],  # True: 2, Top-2: [0, 1] ✗
])

top1_acc = top_k_accuracy(y_true_multi, y_pred_probs, k=1)
top2_acc = top_k_accuracy(y_true_multi, y_pred_probs, k=2)
top3_acc = top_k_accuracy(y_true_multi, y_pred_probs, k=3)

print(f"\nTop-1 Accuracy: {top1_acc:.4f}")
print(f"Top-2 Accuracy: {top2_acc:.4f}")
print(f"Top-3 Accuracy: {top3_acc:.4f}")

# Test multi-class metrics
y_true_multi_class = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
y_pred_multi_class = torch.tensor([0, 1, 2, 1, 1, 2, 0, 2, 1])

p_macro, r_macro, f1_macro = precision_recall_f1(
    y_true_multi_class, y_pred_multi_class, average='macro'
)
print(f"\nMacro Precision: {p_macro:.4f}")
print(f"Macro Recall: {r_macro:.4f}")
print(f"Macro F1: {f1_macro:.4f}")
```

**Common mistakes:**
1. ❌ Confusing rows/columns in confusion matrix
2. ❌ Not handling division by zero
3. ❌ Wrong ROC-AUC calculation (need to sort by scores)
4. ❌ Not considering class imbalance in metrics

## Follow-up Questions
- When to use macro vs micro averaging?
- Why is accuracy misleading for imbalanced datasets?
- How do you interpret a confusion matrix?
