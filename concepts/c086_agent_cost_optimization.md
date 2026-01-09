# Agent Cost and Latency Optimization

**Category:** agents
**Difficulty:** 3
**Tags:** agents, optimization, deployment

## Question
How do you optimize AI agents for cost and latency in production?

## What to Cover
- **Set context by**: Giving a concrete cost breakdown (agents can cost $0.10-$1.00 per task)
- **Must mention**: Cost strategies (model selection per task, caching, context optimization, batch processing, early stopping), latency strategies (parallel execution, streaming, smaller models)
- **Show depth by**: Discussing production metrics to track, cost/quality tradeoffs, and when to optimize (early vs growth vs scale)
- **Avoid**: Only describing optimization techniques without giving concrete numbers or discussing tradeoffs

## Answer
**Problem**: Agents are expensive!
- Many LLM calls per task (10-50+)
- Long context windows
- Tool executions add latency
- Can spiral out of control

**Cost breakdown example:**

```
User task: "Analyze this dataset and create a report"

Agent execution:
1. LLM call (plan): 8k tokens → $0.02
2. Tool: read_file() → free
3. LLM call (analyze): 50k tokens → $0.12
4. Tool: run_analysis() → $0.01 (compute)
5. LLM call (interpret): 20k tokens → $0.05
6. Tool: create_chart() → free
7. LLM call (write report): 30k tokens → $0.07
8. LLM call (review): 40k tokens → $0.10

Total: ~$0.37 per task
At 10k tasks/day: $3,700/day = $110k/month!
```

**Cost optimization strategies:**

**1. Model selection per task**
```python
class SmartAgent:
    def select_model(self, task_type, complexity):
        if task_type == "planning" and complexity == "simple":
            return "gpt-3.5-turbo"  # Cheap, fast

        elif task_type == "analysis" and complexity == "complex":
            return "gpt-4"  # Expensive but capable

        elif task_type == "summary":
            return "claude-haiku"  # Fast, cheap, good at summaries

        return "gpt-4"  # Default

# Example routing
plan = cheap_model.plan(task)  # $0.001
result = expensive_model.execute(plan)  # $0.10
summary = cheap_model.summarize(result)  # $0.002

Total: $0.103 vs $0.30 (3x cheaper!)
```

**2. Caching**
```python
class CachedAgent:
    def __init__(self):
        self.cache = {}  # Or Redis

    def call_llm(self, prompt):
        # Check cache first
        cache_key = hash(prompt)

        if cache_key in self.cache:
            return self.cache[cache_key]  # Free!

        # Make actual call
        result = llm.call(prompt)
        self.cache[cache_key] = result

        return result

# Example
prompt = "Explain what a transformer is"
# First call: $0.01
# Next 100 identical calls: $0 (cached)
```

**Anthropic prompt caching:**
```python
# Cache system prompt and large context
response = anthropic.messages.create(
    model="claude-3-5-sonnet",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": large_doc,  # 50k tokens
                    "cache_control": {"type": "ephemeral"}  # Cache this!
                },
                {
                    "type": "text",
                    "text": "Summarize the above"
                }
            ]
        }
    ]
)

# First call: 50k input tokens = $0.15
# Subsequent calls: 50k cached tokens = $0.015 (10x cheaper!)
```

**3. Context window optimization**
```python
def optimize_context(messages, max_tokens=8000):
    # 1. Remove redundant messages
    messages = remove_duplicates(messages)

    # 2. Summarize old messages
    if count_tokens(messages) > max_tokens:
        old_msgs = messages[:-10]  # Keep last 10
        summary = summarize(old_msgs)  # Compress
        messages = [summary] + messages[-10:]

    # 3. Truncate tool results
    for msg in messages:
        if msg['role'] == 'tool':
            msg['content'] = truncate(msg['content'], max_len=500)

    return messages

# Before: 100k tokens → $0.30
# After: 15k tokens → $0.045 (6.6x cheaper)
```

**4. Batch processing**
```python
# Instead of processing one-by-one
for task in tasks:
    result = agent.run(task)  # 100 tasks × $0.10 = $10

# Batch similar tasks
batched_results = agent.run_batch(tasks)  # $3 total

# How?
# - Single LLM call with all tasks
# - Shared context/planning
# - Parallel tool execution
```

**5. Streaming for perceived latency**
```python
# Without streaming
result = agent.run(task)  # Wait 10 seconds
print(result)  # All at once

# With streaming
for chunk in agent.run_stream(task):
    print(chunk, end='')  # Appears faster to user!

# Actual time: Same
# Perceived time: Much better UX
```

**6. Speculative execution**
```python
# Predict what user will ask next
current_task = "Analyze sales data"

# While processing, pre-compute likely next tasks
future_pool = ThreadPool()
future_pool.submit(lambda: analyze_trends())  # Likely next step
future_pool.submit(lambda: create_visualization())

# If user asks for trends → instant (already computed)
```

**7. Early stopping**
```python
def run_with_early_stop(task, max_cost=0.50):
    total_cost = 0

    for step in agent.plan(task):
        step_cost = estimate_cost(step)

        if total_cost + step_cost > max_cost:
            return "Task exceeds budget, stopping early"

        result = execute(step)
        total_cost += step_cost

    return result
```

**Latency optimization:**

**1. Parallel tool execution**
```python
# Sequential (slow)
weather_sf = get_weather("SF")      # 200ms
weather_ny = get_weather("NY")      # 200ms
weather_la = get_weather("LA")      # 200ms
# Total: 600ms

# Parallel (fast)
results = parallel_execute([
    ("get_weather", "SF"),
    ("get_weather", "NY"),
    ("get_weather", "LA")
])
# Total: 200ms (3x faster!)
```

**2. Model size selection**
```python
latency_requirements = {
    "real_time_chat": "gpt-3.5-turbo",     # 500ms
    "background_task": "gpt-4",             # 3000ms OK
    "batch_process": "gpt-4-turbo-128k"   # 10s OK
}
```

**3. Reduce token count**
```python
# Verbose prompt (slow)
prompt = """
I would like you to please analyze the following
document very carefully and provide a detailed
summary of the key points...
"""  # 500 tokens

# Concise prompt (fast)
prompt = "Summarize key points:"  # 10 tokens

# Same quality, 50x fewer tokens!
```

**4. Prefetching and precomputation**
```python
class PrefetchingAgent:
    def on_user_message(self, msg):
        # Start LLM call immediately
        llm_future = async_call_llm(msg)

        # While LLM is thinking, prefetch data
        if "sales" in msg:
            sales_data = fetch_sales_data()

        # Wait for LLM
        response = llm_future.result()

        return response
```

**Production metrics to track:**

```python
class AgentMetrics:
    def log_task(self, task_id, metrics):
        self.metrics.append({
            'task_id': task_id,
            'cost': metrics['cost'],
            'latency': metrics['latency'],
            'llm_calls': metrics['llm_calls'],
            'tokens': metrics['tokens'],
            'success': metrics['success']
        })

    def get_stats(self):
        return {
            'avg_cost_per_task': mean([m['cost'] for m in self.metrics]),
            'p50_latency': percentile([m['latency'] for m in self.metrics], 50),
            'p95_latency': percentile([m['latency'] for m in self.metrics], 95),
            'success_rate': mean([m['success'] for m in self.metrics]),
            'cost_per_success': sum(costs) / sum(successes)
        }
```

**Real-world optimization examples:**

**Anthropic Claude Projects:**
- Caches uploaded documents (90% cost reduction)
- Haiku for simple tasks, Opus for complex
- Streaming for better UX

**OpenAI Assistants:**
- Function calling (cheaper than full agent loop)
- Persistent threads (context reuse)
- Retrieval only when needed

**Cost/quality tradeoffs:**

```
Scenario: Customer support agent

Option 1: GPT-4 for everything
- Cost: $0.50/conversation
- Quality: 95% satisfaction
- Cost/conversation: $0.50

Option 2: GPT-3.5 → GPT-4 escalation
- 80% handled by GPT-3.5 ($0.05)
- 20% escalated to GPT-4 ($0.50)
- Avg cost: 0.8*$0.05 + 0.2*$0.50 = $0.14
- Quality: 92% satisfaction
- 3.5x cheaper, slightly lower quality

Choose Option 2!
```

**When to optimize:**

Early stage: Focus on capability
- Get it working first
- Don't over-optimize
- Measure baseline costs

Growth stage: Cost matters
- Implement caching
- Model routing
- Monitor per-user costs

Scale: Every cent counts
- A/B test cheaper models
- Aggressive caching
- Custom optimizations

## Follow-up Questions
- How would you implement cost limits per user?
- What's the tradeoff between caching and freshness?
- How do you measure ROI for agent optimization work?
