# Implement Cosine Similarity and Embedding Retrieval

**Category:** coding
**Difficulty:** 2
**Tags:** coding, embeddings, similarity, retrieval, safety

## Question
Implement cosine similarity search over an embedding database. This is used for semantic search, safety classifiers (finding similar unsafe content), content filtering, and RAG systems.

Your implementation should include:
1. **`cosine_similarity`**: Compute pairwise cosine similarity between two sets of vectors
2. **`build_index`**: Create a searchable index from a collection of embeddings
3. **`topk_search`**: Find the k most similar items to a query
4. **`batch_classify`**: Binary classification by comparing to positive/negative exemplars

**Function signature:**
```python
def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between all pairs of vectors in a and b.

    cos_sim(a, b) = (a · b) / (||a|| * ||b||)

    Args:
        a: shape (n, dim)
        b: shape (m, dim)
    Returns:
        similarities: shape (n, m), values in [-1, 1]
    """
    pass

def topk_search(
    query: torch.Tensor,
    database: torch.Tensor,
    k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Find the k most similar vectors in database to the query.

    Args:
        query: shape (dim,) or (num_queries, dim)
        database: shape (num_items, dim)
        k: number of results to return
    Returns:
        scores: shape (num_queries, k) — cosine similarity scores
        indices: shape (num_queries, k) — indices into database
    """
    pass

def batch_classify(
    queries: torch.Tensor,
    positive_exemplars: torch.Tensor,
    negative_exemplars: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Classify queries by comparing average similarity to positive vs negative exemplars.

    This is a simple few-shot classifier using embeddings — useful for
    content safety classification.

    Args:
        queries: shape (num_queries, dim) — items to classify
        positive_exemplars: shape (num_pos, dim) — examples of the positive class
        negative_exemplars: shape (num_neg, dim) — examples of the negative class
    Returns:
        predictions: shape (num_queries,) — True if classified as positive
        confidence: shape (num_queries,) — margin (pos_score - neg_score)
    """
    pass
```

## Answer

**Key concepts:**
1. Cosine similarity measures angle between vectors: 1 = identical direction, 0 = orthogonal, -1 = opposite
2. Normalize vectors first, then cosine similarity = dot product
3. For large databases, pre-normalize and use matrix multiplication for efficiency
4. Few-shot classification: compare query to exemplars from each class

**Reference implementation:**
```python
import torch
import torch.nn.functional as F

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity between rows of a and b."""
    # Normalize each vector to unit length
    a_norm = F.normalize(a, p=2, dim=-1)  # (n, dim)
    b_norm = F.normalize(b, p=2, dim=-1)  # (m, dim)

    # Cosine similarity = dot product of normalized vectors
    return a_norm @ b_norm.T  # (n, m)

def topk_search(
    query: torch.Tensor,
    database: torch.Tensor,
    k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find k most similar items in database."""
    # Handle single query
    if query.dim() == 1:
        query = query.unsqueeze(0)

    # Compute all similarities
    sims = cosine_similarity(query, database)  # (num_queries, num_items)

    # Get top-k
    scores, indices = sims.topk(k, dim=-1)  # both (num_queries, k)

    return scores, indices

def batch_classify(
    queries: torch.Tensor,
    positive_exemplars: torch.Tensor,
    negative_exemplars: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify by comparing similarity to positive vs negative exemplars."""
    # Compute similarity to all exemplars
    pos_sims = cosine_similarity(queries, positive_exemplars)  # (num_queries, num_pos)
    neg_sims = cosine_similarity(queries, negative_exemplars)  # (num_queries, num_neg)

    # Average similarity to each class
    pos_scores = pos_sims.mean(dim=-1)  # (num_queries,)
    neg_scores = neg_sims.mean(dim=-1)  # (num_queries,)

    # Classify based on which class is more similar
    predictions = pos_scores > neg_scores
    confidence = pos_scores - neg_scores

    return predictions, confidence
```

**Testing:**
```python
import torch
import torch.nn.functional as F

torch.manual_seed(1)

# Test 1: Cosine similarity properties
print("=" * 70)
print("TEST 1: Cosine Similarity Properties")
print("=" * 70)
a = torch.tensor([[1.0, 0.0, 0.0]])
b = torch.tensor([
    [1.0, 0.0, 0.0],   # Same direction
    [0.0, 1.0, 0.0],   # Orthogonal
    [-1.0, 0.0, 0.0],  # Opposite
    [0.5, 0.5, 0.0],   # 45 degrees
])

sims = cosine_similarity(a, b)
print(f"Same direction:  {sims[0, 0]:.4f} (should be 1.0)")
print(f"Orthogonal:      {sims[0, 1]:.4f} (should be 0.0)")
print(f"Opposite:        {sims[0, 2]:.4f} (should be -1.0)")
print(f"45 degrees:      {sims[0, 3]:.4f} (should be ~0.707)")

# Test 2: Scale invariance
print("\n" + "=" * 70)
print("TEST 2: Scale Invariance")
print("=" * 70)
v1 = torch.tensor([[3.0, 4.0]])
v2 = torch.tensor([[6.0, 8.0]])   # Same direction, 2x magnitude
v3 = torch.tensor([[0.3, 0.4]])   # Same direction, 0.1x magnitude

sim_12 = cosine_similarity(v1, v2)
sim_13 = cosine_similarity(v1, v3)
print(f"v1 vs 2*v1: {sim_12[0, 0]:.4f} (should be 1.0)")
print(f"v1 vs 0.1*v1: {sim_13[0, 0]:.4f} (should be 1.0)")

# Test 3: Top-k search
print("\n" + "=" * 70)
print("TEST 3: Top-K Search")
print("=" * 70)
database = torch.randn(100, 32)
query = database[42]  # Query IS in the database

scores, indices = topk_search(query, database, k=5)
print(f"Top-1 index: {indices[0, 0].item()} (should be 42)")
print(f"Top-1 score: {scores[0, 0].item():.4f} (should be 1.0)")
print(f"Scores descending: {(scores[0, :-1] >= scores[0, 1:]).all().item()}")

# Test 4: Batch queries
print("\n" + "=" * 70)
print("TEST 4: Batch Queries")
print("=" * 70)
queries = database[:3]  # Use first 3 items as queries
scores, indices = topk_search(queries, database, k=3)
print(f"Scores shape: {scores.shape} (should be (3, 3))")
print(f"Each query finds itself: {(indices[:, 0] == torch.arange(3)).all().item()}")

# Test 5: Few-shot classification
print("\n" + "=" * 70)
print("TEST 5: Embedding-Based Classification")
print("=" * 70)
dim = 16

# Create two clusters
pos_center = torch.randn(dim)
neg_center = torch.randn(dim)

positive_exemplars = pos_center + torch.randn(10, dim) * 0.3
negative_exemplars = neg_center + torch.randn(10, dim) * 0.3

# Queries: 5 positive, 5 negative
queries_pos = pos_center + torch.randn(5, dim) * 0.3
queries_neg = neg_center + torch.randn(5, dim) * 0.3
queries = torch.cat([queries_pos, queries_neg])
true_labels = torch.tensor([True] * 5 + [False] * 5)

predictions, confidence = batch_classify(queries, positive_exemplars, negative_exemplars)
accuracy = (predictions == true_labels).float().mean().item()
print(f"Classification accuracy: {accuracy:.2f}")
print(f"Predictions: {predictions.tolist()}")
print(f"True labels: {true_labels.tolist()}")
print(f"Confidence:  {[f'{c:.3f}' for c in confidence.tolist()]}")

# Test 6: Compare with PyTorch's cosine_similarity
print("\n" + "=" * 70)
print("TEST 6: Verify Against PyTorch")
print("=" * 70)
a = torch.randn(5, 32)
b = torch.randn(5, 32)

# Our pairwise version
ours = cosine_similarity(a, b)

# PyTorch's per-pair version (only works for matching pairs)
pytorch_diag = F.cosine_similarity(a, b, dim=-1)
our_diag = ours.diag()

print(f"Diagonal match: {torch.allclose(our_diag, pytorch_diag, atol=1e-5)}")
```

**Common mistakes:**
1. Not normalizing vectors before dot product
2. Dividing by norm without epsilon (division by zero for zero vectors)
3. Confusing pairwise similarity (n, m) with paired similarity (n,)
4. Using L2 distance instead of cosine similarity (different ranking for different magnitudes)
5. Not handling the 1D query case (single vector vs batch)

## Follow-up Questions
- When would you use L2 distance vs cosine similarity?
- How do you efficiently search over millions of embeddings? (FAISS, approximate NN)
- How are embedding-based classifiers used for content safety?
- What is the relationship between cosine similarity and the dot product attention score?
- Why is cosine similarity scale-invariant, and when is that a problem?
