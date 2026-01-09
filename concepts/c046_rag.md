# Retrieval-Augmented Generation (RAG)

**Category:** modern_llm
**Difficulty:** 3
**Tags:** rag, retrieval, architecture

## Question
What is RAG and when should you use it vs fine-tuning?

## What to Cover
- **Set context by**: Explaining RAG as augmenting generation with retrieved context
- **Must mention**: The pipeline (embed → retrieve → augment prompt → generate), components (retriever, vector DB, generator), RAG vs fine-tuning tradeoffs
- **Show depth by**: Discussing challenges (retrieval quality, context length limits) and advanced techniques (reranking, HyDE)
- **Avoid**: Only describing the pipeline without explaining when to use RAG vs fine-tuning

## Answer
**RAG**: Retrieve relevant documents and include in context when generating.

**Pipeline:**
```
1. User query: "What is the capital of France?"

2. Retrieval:
   - Embed query with retrieval model (e.g., Sentence-BERT)
   - Search vector database for similar documents
   - Return top-k documents (k=3-10)

3. Augment prompt:
   """
   Given the following context:
   [Retrieved doc 1]
   [Retrieved doc 2]
   ...

   Answer: What is the capital of France?
   """

4. Generate answer using LLM
```

**Components:**

**Retriever:**
- **Dense**: Embedding models (BERT, Sentence-T5)
- **Sparse**: BM25, TF-IDF (keyword-based)
- **Hybrid**: Combine dense + sparse

**Vector Database:**
- FAISS, Pinecone, Weaviate, Chroma
- Store embeddings, fast similarity search

**Generator:**
- Any LLM (GPT-4, Claude, open-source)

**RAG vs Fine-tuning:**

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| Knowledge update | Easy (update DB) | Expensive (retrain) |
| Factual grounding | Strong (cites sources) | Weak (hallucinates) |
| Customization | Limited (retrieval only) | Full (behavior change) |
| Latency | Higher (retrieval step) | Lower (direct generation) |
| Cost | Inference only | Training compute |

**When to use RAG:**

✓ Frequently changing information (news, docs)
✓ Need citations/sources
✓ Large knowledge base (can't fit in training)
✓ Domain-specific (internal docs)

**When to use fine-tuning:**

✓ Change style/format/behavior
✓ Domain-specific language patterns
✓ Low latency requirements
✓ Small, static knowledge base

**Often best: RAG + fine-tuning**
- Fine-tune for task format/style
- RAG for knowledge retrieval

**Challenges:**

1. **Retrieval quality**: Wrong docs → wrong answer
2. **Context length**: Limited tokens for retrieved docs
3. **Ranking**: Which docs to include?
4. **Cost**: Embedding + retrieval + generation

**Advanced RAG:**
- **Reranking**: Use LLM to rerank retrieved docs
- **Iterative retrieval**: Generate → retrieve more → refine
- **Hypothetical document embeddings (HyDE)**: Generate answer first, then retrieve

## Follow-up Questions
- How do you evaluate RAG quality?
- What embedding model should you use for retrieval?
- How many documents should you retrieve?
