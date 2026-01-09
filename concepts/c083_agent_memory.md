# Agent Memory Systems

**Category:** agents
**Difficulty:** 3
**Tags:** agents, memory, architecture

## Question
How do AI agents handle memory? What are the different memory architectures?

## What to Cover
- **Set context by**: Explaining the problem (LLMs have fixed context, agents need long-term memory)
- **Must mention**: Memory types (short-term/working, long-term/episodic, entity/semantic, procedural/skills), implementation approaches (vector DB, summarization)
- **Show depth by**: Discussing optimization strategies (summarization, compression, forgetting) and challenges (what to remember, privacy, consistency)
- **Avoid**: Only describing memory types without explaining how they're implemented and the challenges

## Answer
**Problem**: LLMs have fixed context windows, but agents need to remember:
- Past conversations
- Learned facts
- Tool results
- Long-term user preferences

**Memory types:**

**1. Short-term (working) memory**
```
= Current conversation context
= What fits in the prompt

Example (8k context):
User: "Book me a flight to Paris"
Agent: [searches flights]
User: "The cheapest one"  ← Needs to remember flight options
Agent: [books flight from previous results]
```

**Implementation:**
```python
class ShortTermMemory:
    def __init__(self, max_tokens=8000):
        self.messages = []
        self.max_tokens = max_tokens

    def add(self, message):
        self.messages.append(message)

        # Truncate if too long
        while self.count_tokens() > self.max_tokens:
            # Remove oldest non-system messages
            self.messages.pop(1)  # Keep system prompt

    def get_context(self):
        return "\n".join(self.messages)
```

**2. Long-term memory (episodic)**
```
= Vector database of past interactions
= Semantic search to retrieve relevant memories

Example:
User (today): "What restaurants did I like?"
→ Search memory for past restaurant discussions
→ Retrieve: "You enjoyed Sushi Place X last month"
```

**Implementation:**
```python
class LongTermMemory:
    def __init__(self):
        self.vector_db = VectorDB()  # Pinecone, Weaviate, etc.

    def store(self, conversation, metadata):
        embedding = embed(conversation)
        self.vector_db.upsert(
            id=metadata['timestamp'],
            vector=embedding,
            metadata={
                'text': conversation,
                'user_id': metadata['user_id'],
                'timestamp': metadata['timestamp']
            }
        )

    def retrieve(self, query, k=5):
        query_embedding = embed(query)
        results = self.vector_db.search(
            vector=query_embedding,
            top_k=k
        )
        return [r['metadata']['text'] for r in results]
```

**3. Entity memory (semantic)**
```
= Structured facts about entities
= Knowledge graph or database

Example:
Entity: "John Smith"
Facts:
  - Works at Google
  - Lives in SF
  - Prefers vegetarian restaurants
  - Last interaction: 2024-01-05
```

**Implementation:**
```python
class EntityMemory:
    def __init__(self):
        self.entities = {}  # Could be a graph DB

    def update_entity(self, name, facts):
        if name not in self.entities:
            self.entities[name] = {}

        self.entities[name].update(facts)
        self.entities[name]['last_updated'] = datetime.now()

    def get_entity(self, name):
        return self.entities.get(name, {})

# Usage
entity_mem.update_entity("John", {
    "employer": "Google",
    "dietary_preference": "vegetarian"
})

# Later...
john_info = entity_mem.get_entity("John")
# Use in prompt: "Remember John prefers vegetarian food"
```

**4. Procedural memory (skills)**
```
= Learned procedures and code snippets
= Reusable functions the agent has created

Example (Voyager):
skills = {
    "mine_wood": "function() { ... }",
    "craft_pickaxe": "function() { ... }",
    "find_cave": "function() { ... }"
}
```

**Full memory architecture:**

```python
class AgentMemory:
    def __init__(self):
        self.short_term = ShortTermMemory(max_tokens=8000)
        self.long_term = LongTermMemory()
        self.entities = EntityMemory()
        self.skills = SkillMemory()

    def process_interaction(self, user_msg, agent_response):
        # 1. Add to short-term
        self.short_term.add(user_msg)
        self.short_term.add(agent_response)

        # 2. Extract and store entities
        entities = extract_entities(user_msg)
        for entity, facts in entities.items():
            self.entities.update_entity(entity, facts)

        # 3. Store in long-term if important
        if is_important(user_msg, agent_response):
            self.long_term.store(
                conversation=f"User: {user_msg}\nAgent: {agent_response}",
                metadata={'timestamp': datetime.now()}
            )

    def get_context_for_query(self, query):
        context = []

        # Short-term memory (always included)
        context.append(self.short_term.get_context())

        # Relevant long-term memories
        relevant_memories = self.long_term.retrieve(query, k=3)
        if relevant_memories:
            context.append("Relevant past conversations:")
            context.extend(relevant_memories)

        # Relevant entities
        mentioned_entities = extract_entities(query)
        for entity in mentioned_entities:
            info = self.entities.get_entity(entity)
            if info:
                context.append(f"About {entity}: {info}")

        return "\n\n".join(context)
```

**Production examples:**

**ChatGPT with Memory:**
```
Stores:
- User preferences ("I prefer Python over JavaScript")
- Personal facts ("I live in Seattle")
- Context from previous chats (opt-in)

Retrieval:
- Semantic search on new query
- Top 3-5 relevant memories added to context
```

**Claude Projects:**
```
Short-term: Conversation in current chat
Long-term: Project knowledge (uploaded docs)
Retrieval: Semantic search over project docs
```

**Memory optimization strategies:**

**1. Summarization:**
```python
# When context gets too long, summarize old messages
if self.short_term.count_tokens() > 6000:
    old_messages = self.short_term.messages[:10]
    summary = llm.summarize(old_messages)

    # Replace old messages with summary
    self.short_term.messages = [
        system_prompt,
        {"role": "system", "content": f"Previous context: {summary}"}
    ] + self.short_term.messages[10:]
```

**2. Compression:**
```python
# Compress tool results
tool_result = api.search_web(query)  # Returns 50 results

# Don't store all 50, compress to top 5
compressed = tool_result[:5]
memory.add(f"Search results (top 5): {compressed}")
```

**3. Forgetting:**
```python
# Decay old memories
def retrieve_with_recency(query, k=5):
    results = vector_search(query, k=20)  # Get more candidates

    # Re-rank by relevance + recency
    scored = [
        (r, similarity_score(r) * recency_weight(r))
        for r in results
    ]

    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]

def recency_weight(memory):
    days_old = (datetime.now() - memory.timestamp).days
    return 1.0 / (1.0 + days_old / 30)  # Decay over ~month
```

**Challenges:**

**1. What to remember?**
```
❌ "The weather is sunny"  ← Outdated quickly
✓ "User prefers outdoor activities"  ← Long-term preference
```

**2. Privacy:**
```
Problem: Storing personal information
Solution:
- User controls (view, delete memories)
- Encryption at rest
- Data retention policies
```

**3. Consistency:**
```
Problem:
Memory 1: "User lives in SF"
Memory 2: "User moved to NYC"

Solution:
- Timestamp-based resolution
- Conflict detection and resolution
- Entity versioning
```

**4. Retrieval quality:**
```
Problem: Retrieve irrelevant memories
Solution:
- Hybrid search (keyword + semantic)
- Reranking
- User feedback on relevance
```

**Cost considerations:**

```
Vector DB costs:
- Storage: $0.25/GB/month (Pinecone)
- Queries: $0.10/million queries

For 1M users:
- 100 memories each = 100M vectors
- ~500GB storage = $125/month
- 10M queries/day = $30/day = $900/month

Total: ~$1k/month
```

## Follow-up Questions
- How would you handle conflicting information in memory?
- What's the tradeoff between retrieval quality and cost?
- How does memory affect agent behavior over time?
