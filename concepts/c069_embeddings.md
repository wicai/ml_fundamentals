# Text Embeddings

**Category:** modern_llm
**Difficulty:** 3
**Tags:** embeddings, retrieval, representation

## Question
What are text embeddings and how are they used for semantic search and RAG?

## Answer
**Text Embedding**: Dense vector representation of text that captures semantic meaning.

**Goal**: Similar texts → similar vectors (measured by cosine similarity or dot product).

**Example:**
```
"cat" → [0.2, 0.8, -0.3, ...] (512-d vector)
"kitten" → [0.22, 0.78, -0.28, ...] (similar!)
"car" → [-0.5, 0.1, 0.9, ...] (different)

cosine_sim("cat", "kitten") = 0.95
cosine_sim("cat", "car") = 0.12
```

**How embeddings are trained:**

**1. Contrastive learning (e.g., Sentence-BERT):**
```
Positive pairs: (query, relevant doc)
Negative pairs: (query, irrelevant doc)

Loss: Maximize sim(query, relevant), minimize sim(query, irrelevant)

Triplet loss: d(anchor, pos) < d(anchor, neg) + margin
```

**2. Supervised fine-tuning:**
```
Start: Pretrained model (BERT)
Fine-tune: On (question, answer) pairs for similarity

Data: NLI datasets (SNLI, MNLI), QA datasets
```

**3. Unsupervised (masked LM, next sentence prediction):**
```
Use [CLS] token or mean pooling
Doesn't capture similarity as well
```

**Popular embedding models:**

| Model | Dimensions | Use case | Performance |
|-------|------------|----------|-------------|
| text-embedding-ada-002 (OpenAI) | 1536 | General | Excellent |
| all-MiniLM-L6-v2 | 384 | Fast, small | Good |
| e5-large-v2 | 1024 | General | Excellent |
| Sentence-T5 | 768 | General | Great |
| BGE-large-en | 1024 | Retrieval | State-of-art |
| Cohere Embed | 1024 | General | Excellent |

**Applications:**

**1. Semantic search:**
```
1. Embed all documents offline
2. Store in vector DB
3. User query → embed query
4. Find nearest neighbors (cosine similarity)
5. Return top-k documents
```

**2. RAG (Retrieval-Augmented Generation):**
```
Query: "What is photosynthesis?"
  1. Embed query
  2. Retrieve relevant docs from vector DB
  3. Pass docs + query to LLM
  4. Generate answer grounded in docs
```

**3. Clustering:**
```
Embed documents → cluster embeddings → topic groups
```

**4. Recommendation:**
```
User likes doc A → find similar docs via embeddings
```

**5. Deduplication:**
```
Find near-duplicate documents (high similarity)
```

**Vector databases:**

- **FAISS**: Fast approximate search (Facebook)
- **Pinecone**: Managed vector DB
- **Weaviate**: GraphQL vector search
- **Chroma**: Lightweight, open-source
- **Qdrant**: Rust-based, fast
- **Milvus**: Distributed vector DB

**Key metrics:**

**Similarity measures:**
```
Cosine similarity: dot(a, b) / (||a|| * ||b||)
  Range: [-1, 1], higher = more similar

Euclidean distance: ||a - b||
  Lower = more similar

Dot product: dot(a, b)
  Faster, assumes normalized vectors
```

**Retrieval metrics:**
- Recall@k: Fraction of relevant docs in top-k
- MRR (Mean Reciprocal Rank): 1 / rank of first relevant doc
- NDCG: Considers ranking quality

**Best practices:**

**1. Normalize embeddings:**
```
emb = emb / ||emb||  # Unit length
Allows dot product = cosine similarity (faster)
```

**2. Batch encoding:**
```
Encode 100 texts at once (not one-by-one)
Much faster with GPUs
```

**3. Dimensionality:**
- Higher dims (1024+): Better quality, more memory
- Lower dims (384): Faster, smaller
- Trade-off based on use case

**4. Domain adaptation:**
```
Fine-tune embedding model on your domain
(medical, legal, code, etc.)
Improves relevance
```

**Challenges:**

1. **Out-of-domain**: Model trained on general text, used on code → poor
2. **Short queries**: "python list" → many matches
3. **Semantic gap**: Query != how answer is phrased
4. **Language**: Most models English-only or multilingual with degradation

**Modern trends:**

- **Matryoshka embeddings**: One model, multiple dimensions (truncate 1024 → 512 → 256)
- **Late interaction (ColBERT)**: Token-level embeddings, more precise
- **Learned sparse retrieval**: Sparse + dense hybrid

## Follow-up Questions
- How are embeddings different from word2vec?
- What's the difference between symmetric and asymmetric embedding models?
- How do you choose embedding dimensionality?
