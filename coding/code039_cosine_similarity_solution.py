# Implement Cosine Similarity and Embedding Retrieval
# ====================================================================
#
# Implement cosine similarity search over an embedding database. This is used for semantic search, safety classifiers (finding similar unsafe content), content filtering, and RAG systems.
# 
# Your implementation should include:
# 1. **`cosine_similarity`**: Compute pairwise cosine similarity between two sets of vectors
# 2. **`build_index`**: Create a searchable index from a collection of embeddings
# 3. **`topk_search`**: Find the k most similar items to a query
# 4. **`batch_classify`**: Binary classification by comparing to positive/negative exemplars
# 
# **Function signature:**
#
# ====================================================================
import torch
from torch import nn
from torch.nn import functional as F
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
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.T

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
    if len(query.shape) == 1:
        query = query.unsqueeze(0)        
    # now it's (num_queries, dim)
    # for each query I wanna know the closest items in database to it 
    similarities = cosine_similarity(query, database) # (num_queries, num_items)
    top_k = torch.topk(similiarities, k, dim=-1)
    return top_k.values, top_k.indices    


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
    pos_sim = cosine_similarity(queries, positive_exemplars) #num_queries, num_pos
    neg_sim = cosine_similarity(queries, negative_exemplars) #num_queries, num_neg
    avg_pos_dists = pos_sim.mean(dim=-1)
    avg_neg_dists = neg_sim.mean(dim=-1)
    diff = avg_pos_dists - avg_neg_dists #num_queries,
    predictions = (diff > 0)
    return (predictions, diff)

