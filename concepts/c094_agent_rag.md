# Retrieval for Agents (RAG)

**Category:** agents
**Difficulty:** 3
**Tags:** agents, rag, retrieval

## Question
How do agents use retrieval (RAG)? What are the key patterns and challenges?

## What to Cover
- **Set context by**: Explaining how agent RAG differs from simple chat RAG (agents decide when and what to retrieve)
- **Must mention**: Retrieval patterns (single, iterative, multi-query, agentic tool-based), advanced techniques (HyDE, Self-RAG), challenges (query formulation, relevance filtering, citation, handling no results)
- **Show depth by**: Discussing production patterns and metrics for evaluating RAG quality in agents
- **Avoid**: Only describing basic RAG without explaining agent-specific retrieval patterns and challenges

## Answer
**Retrieval for agents** = Agent fetches relevant information to ground responses.

**Why retrieval for agents:**
- Access up-to-date information
- Ground in facts (reduce hallucination)
- Answer from private documents
- Scale beyond context window

**Agent RAG vs Chat RAG:**

**Chat RAG (simpler):**
```
User question → Retrieve docs → LLM → Answer
```

**Agent RAG (more complex):**
```
Agent plans query → Retrieve → Agent analyzes → Agent decides next action
→ May retrieve multiple times with different queries
```

**Retrieval patterns:**

**1. Single retrieval**
```python
def simple_agent_rag(question):
    # Retrieve once
    docs = retrieve(question, k=5)

    # Answer based on docs
    answer = llm(f"""
    Question: {question}

    Context: {docs}

    Answer:
    """)

    return answer
```

**2. Iterative retrieval**
```python
def iterative_retrieval_agent(question):
    answer = ""
    max_iterations = 3

    for i in range(max_iterations):
        # Generate query based on what we know so far
        query = llm(f"""
        Original question: {question}
        What we know: {answer}

        What should we search for next?
        """)

        # Retrieve
        docs = retrieve(query, k=3)

        # Update answer
        answer = llm(f"""
        Question: {question}
        Previous answer: {answer}
        New information: {docs}

        Updated answer:
        """)

        # Check if we have enough info
        if is_complete(answer):
            break

    return answer
```

**3. Multi-query retrieval**
```python
def multi_query_agent(question):
    # Generate multiple search queries
    queries = llm(f"""
    Generate 3 different search queries for: {question}

    Output as list:
    """)

    # Retrieve for each
    all_docs = []
    for query in queries:
        docs = retrieve(query, k=3)
        all_docs.extend(docs)

    # Deduplicate and rerank
    docs = deduplicate_and_rerank(all_docs)

    # Answer
    return llm(f"Question: {question}\nContext: {docs}\n\nAnswer:")
```

**4. Agentic RAG (tool-based)**
```python
tools = [
    {
        "name": "search_docs",
        "description": "Search internal documentation",
        "function": search_docs
    },
    {
        "name": "search_web",
        "description": "Search the web for current information",
        "function": search_web
    }
]

agent = create_agent(tools)

# Agent decides when and what to retrieve
agent.run("What's our company's vacation policy and how does it compare to industry standards?")

# Agent's decisions:
# 1. search_docs("vacation policy") → Gets company policy
# 2. search_web("tech industry vacation days 2024") → Gets industry data
# 3. Compares and answers
```

**Advanced patterns:**

**Hypothetical document embeddings (HyDE):**
```python
# Instead of searching with question, search with hypothetical answer
question = "What is RAG?"

# Generate hypothetical answer
hypothetical = llm(f"Answer this question: {question}")
# → "RAG is retrieval augmented generation..."

# Search with hypothetical answer (often better retrieval)
docs = retrieve(hypothetical, k=5)

# Generate real answer
answer = llm(f"Question: {question}\nContext: {docs}")
```

**Self-RAG (agent decides when to retrieve):**
```python
def self_rag_agent(question):
    # Agent decides if retrieval needed
    needs_retrieval = llm(f"""
    Question: {question}

    Do you need to retrieve information? (yes/no)
    """)

    if needs_retrieval == "yes":
        docs = retrieve(question)
        answer = llm(f"Question: {question}\nContext: {docs}")
    else:
        # Answer from parametric knowledge
        answer = llm(f"Question: {question}")

    return answer
```

**Retrieval challenges for agents:**

**1. Query formulation**
```python
# User question
"What did John say about the Q3 numbers?"

# Bad retrieval query (too vague)
retrieve("John Q3 numbers")

# Good query (more specific)
retrieve("John Q3 financial results revenue")
```

**2. Relevance filtering**
```python
# Retrieved docs may not all be relevant
docs = retrieve(query, k=10)

# Agent filters
relevant_docs = []
for doc in docs:
    relevance = llm(f"""
    Query: {query}
    Document: {doc}

    Is this document relevant? (yes/no)
    """)

    if relevance == "yes":
        relevant_docs.append(doc)

# Use only relevant docs
```

**3. Citation and grounding**
```python
answer = llm(f"""
Question: {question}
Context: {docs}

Answer the question and cite which documents you used.
""")

# Example output:
# "Based on Document 2, the revenue was $10M [2].
#  Document 1 mentions similar figures [1]."
```

**4. Handling no results**
```python
docs = retrieve(query)

if len(docs) == 0:
    # No results, try alternative query
    alternative_query = rephrase(query)
    docs = retrieve(alternative_query)

    if len(docs) == 0:
        return "I don't have information about that in my knowledge base"
```

**Production patterns:**

**Claude with retrieval:**
```python
# Add retrieved docs to context
context = retrieve_docs(user_question)

response = anthropic.messages.create(
    model="claude-3-5-sonnet",
    messages=[{
        "role": "user",
        "content": f"""
        Context: {context}

        Question: {user_question}

        Answer based only on the context above.
        """
    }]
)
```

**OpenAI Assistants with retrieval:**
```python
# Upload files
file = client.files.create(file=open("docs.pdf"), purpose="assistants")

# Create assistant with retrieval
assistant = client.beta.assistants.create(
    model="gpt-4-turbo",
    tools=[{"type": "retrieval"}],
    file_ids=[file.id]
)

# Assistant automatically retrieves from uploaded files
```

**Metrics:**

```python
rag_metrics = {
    "retrieval_precision": 0.85,  # % of retrieved docs relevant
    "retrieval_recall": 0.70,     # % of relevant docs retrieved
    "answer_accuracy": 0.90,      # % of answers correct
    "citation_accuracy": 0.80,    # % of citations correct
    "avg_retrieval_time": 200,    # ms
}
```

## Follow-up Questions
- How do you evaluate retrieval quality for agents?
- When should agents retrieve vs use parametric knowledge?
- How do you handle conflicting information across retrieved docs?
