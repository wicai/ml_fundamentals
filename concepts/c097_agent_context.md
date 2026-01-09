# Agent Context Window Management

**Category:** agents
**Difficulty:** 3
**Tags:** agents, context, optimization

## Question
How do agents manage limited context windows? What strategies optimize context usage?

## What to Cover
- **Set context by**: Explaining the context problem (system prompt + history + tools + retrieved docs = overflow)
- **Must mention**: Context management strategies (summarization, sliding window, hierarchical compression, selective inclusion, tool result compression, caching, retrieval offloading), token budget allocation, adaptive compression
- **Show depth by**: Discussing tradeoffs between strategies and how long-context models change the calculus
- **Avoid**: Only describing one strategy without comparing approaches and their tradeoffs

## Answer
**The context problem:**

```
Agent with 8k context window:
- System prompt: 1k tokens
- Conversation history: 3k tokens
- Tool descriptions: 1k tokens
- Retrieved docs: 2k tokens
- Working room: 1k tokens ← Getting tight!

Next step requires 2k tokens → Overflow!
```

**Context management strategies:**

**1. Summarization**
```python
class SummarizingAgent:
    def __init__(self, max_tokens=8000):
        self.messages = []
        self.max_tokens = max_tokens

    def add_message(self, msg):
        self.messages.append(msg)

        # Check if over budget
        if self.count_tokens() > self.max_tokens * 0.8:
            # Summarize old messages
            old_messages = self.messages[:-10]  # Keep recent 10
            summary = self.summarize(old_messages)

            # Replace with summary
            self.messages = [
                {"role": "system", "content": f"Summary of earlier conversation: {summary}"}
            ] + self.messages[-10:]
```

**2. Sliding window**
```python
class SlidingWindowAgent:
    def get_context(self):
        # Keep only recent messages
        recent_messages = self.messages[-20:]  # Last 20 messages

        # Always include system prompt
        context = [self.system_prompt] + recent_messages

        return context
```

**3. Hierarchical compression**
```python
class HierarchicalMemory:
    def __init__(self):
        self.hot = []   # Last 5 messages (full)
        self.warm = []  # Last 20 messages (summarized)
        self.cold = []  # Older (highly compressed)

    def get_context(self):
        return {
            "recent": self.hot,           # 1k tokens
            "summary": self.warm[0],      # 500 tokens
            "background": self.cold[0]    # 200 tokens
        }
```

**4. Selective inclusion**
```python
def get_relevant_context(current_task, full_history):
    # Score each message by relevance
    scored = []
    for msg in full_history:
        relevance = compute_relevance(msg, current_task)
        scored.append((msg, relevance))

    # Sort by relevance
    scored.sort(key=lambda x: x[1], reverse=True)

    # Take top K most relevant
    relevant = [msg for msg, score in scored[:10]]

    return relevant
```

**5. Tool result compression**
```python
def compress_tool_result(result):
    # Tool returns 10k tokens of data
    if len(result) > 1000:
        # Compress to key points
        compressed = llm(f"""
        Summarize this in 200 tokens:
        {result}
        """)
        return compressed

    return result

# Example
search_results = search("AI agents")  # 50 results, 5k tokens
compressed = compress_tool_result(search_results)  # Top 5 results, 500 tokens
```

**6. Caching (Anthropic prompt caching)**
```python
# Cache expensive context
response = anthropic.messages.create(
    model="claude-3-5-sonnet",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": large_doc,  # 50k tokens
                "cache_control": {"type": "ephemeral"}  # Cache this!
            },
            {
                "type": "text",
                "text": "Question about the doc"
            }
        ]
    }]
)

# First call: Full price
# Subsequent calls: 90% cheaper (cached input)
```

**7. Offloading to retrieval**
```python
# Instead of keeping everything in context
# Store in vector DB, retrieve when needed

class RetrievalBackedAgent:
    def __init__(self):
        self.vector_db = VectorDB()
        self.short_context = []  # Only recent messages

    def process(self, user_msg):
        # Store in long-term memory
        self.vector_db.add(user_msg)

        # Retrieve relevant history
        relevant = self.vector_db.search(user_msg, k=3)

        # Build minimal context
        context = self.short_context + relevant

        # Generate response
        response = llm(context)

        return response
```

**Token budget allocation:**

```python
class TokenBudgetManager:
    def __init__(self, max_tokens=8000):
        self.budget = {
            "system_prompt": 1000,      # 12.5%
            "task_description": 500,     # 6.25%
            "conversation": 3000,        # 37.5%
            "tool_results": 2000,        # 25%
            "working_space": 1500        # 18.75%
        }

    def allocate(self, component, tokens):
        if tokens > self.budget[component]:
            # Compress to fit budget
            return self.compress(tokens, target=self.budget[component])

        return tokens
```

**Adaptive compression:**

```python
class AdaptiveAgent:
    def get_context(self, current_task):
        total_tokens = self.count_all_tokens()

        if total_tokens < 4000:
            # Plenty of room, include everything
            compression_level = "none"

        elif total_tokens < 7000:
            # Getting tight, light compression
            compression_level = "light"

        else:
            # Very tight, aggressive compression
            compression_level = "aggressive"

        return self.build_context(compression_level)

    def build_context(self, level):
        if level == "none":
            return self.full_context()

        elif level == "light":
            # Summarize older messages
            return self.summarize_old_messages()

        elif level == "aggressive":
            # Keep only essentials
            return {
                "system": self.system_prompt,
                "current_task": self.current_task,
                "last_3_messages": self.messages[-3:]
            }
```

**Long context models (>100k):**

```python
# With Claude 3 (200k context) or GPT-4 Turbo (128k)
# Can be less aggressive about compression

class LongContextAgent:
    def __init__(self, max_tokens=100_000):
        self.max_tokens = max_tokens
        # Can keep full conversation history
        self.messages = []

    def add_message(self, msg):
        self.messages.append(msg)

        # Only compress when truly necessary
        if self.count_tokens() > self.max_tokens * 0.9:
            self.compress_oldest_quarter()
```

**Trade-offs:**

| Strategy | Pros | Cons |
|----------|------|------|
| Summarization | Preserves key info | Loses details |
| Sliding window | Simple, fast | Forgets old context |
| Selective inclusion | Keeps relevant | Complex, requires scoring |
| Caching | Cost savings | Only for static content |
| Retrieval | Unlimited history | Retrieval latency |

**Context-aware prompting:**

```python
# Adjust prompt based on available context
if context_tokens < 2000:
    # Plenty of room, detailed prompt
    prompt = detailed_system_prompt

elif context_tokens < 6000:
    # Moderate room, concise prompt
    prompt = concise_system_prompt

else:
    # Very limited, minimal prompt
    prompt = minimal_system_prompt
```

**Measuring context efficiency:**

```python
metrics = {
    "avg_context_tokens": 5200,
    "context_utilization": 0.65,  # 65% of max context used
    "compression_rate": 0.30,     # Compressed to 30% of original
    "retrieval_accuracy": 0.85,   # 85% of retrieved context relevant
}
```

**Best practices:**

1. **Monitor token usage**: Track context size over time
2. **Compress early**: Don't wait until overflow
3. **Keep system prompt short**: Every token counts
4. **Truncate tool results**: 500 tokens usually enough
5. **Use caching**: For static content (docs, system prompts)
6. **Retrieval for history**: Don't keep everything in context

## Follow-up Questions
- How do you decide what to compress vs keep in full?
- What's the impact of summarization on agent performance?
- When does retrieval outperform keeping things in context?
