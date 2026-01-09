# Agent Monitoring and Debugging

**Category:** agents
**Difficulty:** 3
**Tags:** agents, monitoring, debugging

## Question
How do you monitor and debug AI agents in production? What metrics and tools do you use?

## What to Cover
- **Set context by**: Explaining why agents are hard to debug (non-deterministic, multi-step, complex tool interactions)
- **Must mention**: Key metrics (success rate, latency, cost, error types), logging architecture (structured logs, trace visualization), debugging techniques (replay, step-by-step inspection), and monitoring tools (LangSmith, Langfuse, W&B)
- **Show depth by**: Discussing alerting strategies, common debugging scenarios (loops, hallucinated tools, cost spikes), and production best practices
- **Avoid**: Only mentioning tools without explaining what metrics to track and how to debug failures

## Answer
**Challenge**: Agents are hard to debug because:
- Non-deterministic behavior
- Multi-step execution
- Complex tool interactions
- Long-running tasks

**Key metrics to monitor:**

**1. Task-level metrics**
```python
{
    "task_id": "uuid",
    "success": true,
    "latency_ms": 3420,
    "cost_usd": 0.15,
    "num_steps": 7,
    "num_llm_calls": 5,
    "num_tool_calls": 3,
    "error": null,
    "user_feedback": "helpful"
}
```

**2. System-level metrics**
```python
# Track over time
metrics = {
    "success_rate": 0.87,  # 87% of tasks succeed
    "avg_cost_per_task": 0.12,
    "p50_latency": 2100,
    "p95_latency": 8500,
    "daily_active_users": 1250,
    "total_cost_today": 450.00
}
```

**3. Error rates by type**
```python
error_breakdown = {
    "timeout": 0.05,           # 5% of failures
    "tool_error": 0.03,        # 3%
    "hallucinated_action": 0.02,  # 2%
    "infinite_loop": 0.01,     # 1%
    "llm_error": 0.02          # 2%
}
```

**Logging architecture:**

**1. Structured logging**
```python
import structlog

logger = structlog.get_logger()

class MonitoredAgent:
    def run(self, task):
        task_id = generate_id()

        logger.info("task_started",
            task_id=task_id,
            task=task,
            user_id=self.user_id
        )

        try:
            for step_num, step in enumerate(self.plan(task)):
                logger.info("step_started",
                    task_id=task_id,
                    step_num=step_num,
                    step=step
                )

                result = self.execute_step(step)

                logger.info("step_completed",
                    task_id=task_id,
                    step_num=step_num,
                    result=result,
                    cost=step.cost
                )

            logger.info("task_completed",
                task_id=task_id,
                success=True,
                total_cost=self.total_cost
            )

        except Exception as e:
            logger.error("task_failed",
                task_id=task_id,
                error=str(e),
                traceback=traceback.format_exc()
            )
```

**2. Trace visualization**
```python
# LangSmith / LangFuse style tracing
{
    "task_id": "123",
    "steps": [
        {
            "name": "planning",
            "type": "llm",
            "start": "2024-01-01T10:00:00",
            "end": "2024-01-01T10:00:02",
            "input": "Plan how to book flight",
            "output": "1. Search flights\n2. Compare prices\n3. Book",
            "cost": 0.02,
            "tokens": {
                "input": 100,
                "output": 50
            }
        },
        {
            "name": "search_flights",
            "type": "tool",
            "start": "2024-01-01T10:00:02",
            "end": "2024-01-01T10:00:03",
            "input": {"dest": "NYC", "date": "2024-02-01"},
            "output": "[flight options...]",
            "cost": 0.00
        },
        {
            "name": "booking",
            "type": "llm",
            "input": "Book the cheapest flight from results",
            "output": "Calling book_flight(...)",
            "cost": 0.01
        }
    ],
    "total_duration_ms": 3200,
    "total_cost": 0.03
}
```

**3. Real-time dashboards**
```
Grafana / Datadog dashboard:

┌─ Success Rate (Last Hour) ─┐
│        87.3%                │
│   ▁▂▃▅▆▇█▇▆▅▃▂▁            │
└─────────────────────────────┘

┌─ Avg Cost/Task ────────────┐
│        $0.12                │
│   Target: < $0.15           │
└─────────────────────────────┘

┌─ Active Errors ────────────┐
│  TimeoutError: 12           │
│  ToolError: 5               │
│  HallucinatedAction: 2      │
└─────────────────────────────┘
```

**Debugging techniques:**

**1. Replay failed tasks**
```python
class ReplayableAgent:
    def run(self, task):
        # Save all inputs/outputs
        self.recorder.start(task)

        try:
            result = self.execute(task)
            self.recorder.save_success(result)
        except Exception as e:
            self.recorder.save_failure(e)
            raise

    def replay(self, task_id):
        # Load saved execution
        recording = self.recorder.load(task_id)

        # Re-run with same inputs
        for step in recording.steps:
            # Can add breakpoints, inspect state, etc.
            result = self.execute_step(step)

            # Compare to original
            assert result == recording.results[step]
```

**2. Step-by-step inspection**
```python
# Debug mode: Manual step-through
agent = DebugAgent(task)

while not agent.done:
    next_action = agent.get_next_action()

    print(f"Next action: {next_action}")
    print(f"Current state: {agent.state}")
    print(f"Cost so far: {agent.cost}")

    # Pause for inspection
    input("Press ENTER to continue...")

    agent.execute_action(next_action)
```

**3. Counterfactual analysis**
```python
# What if we used a different model?
original_result = agent.run(task, model="gpt-4")

# Try with cheaper model
alternative_result = agent.run(task, model="gpt-3.5")

# Compare
print(f"GPT-4: Success={original_result.success}, Cost=${original_result.cost}")
print(f"GPT-3.5: Success={alternative_result.success}, Cost=${alternative_result.cost}")

# Decision: Use 3.5 if success rate acceptable
```

**Alerting:**

```python
class AgentMonitor:
    def check_health(self):
        metrics = self.get_recent_metrics()

        # Alert on degraded performance
        if metrics['success_rate'] < 0.80:
            alert("Success rate dropped to {metrics['success_rate']}")

        # Alert on cost spikes
        if metrics['hourly_cost'] > 100:
            alert(f"Cost spike: ${metrics['hourly_cost']}/hour")

        # Alert on latency
        if metrics['p95_latency'] > 10000:
            alert(f"High latency: {metrics['p95_latency']}ms")

        # Alert on errors
        if metrics['error_rate'] > 0.10:
            alert(f"Error rate: {metrics['error_rate']}")
```

**Common debugging scenarios:**

**1. Agent stuck in loop**
```
Logs show:
  10:00:01 - search("query")
  10:00:02 - No results found
  10:00:03 - search("query")  ← Same query!
  10:00:04 - No results found
  10:00:05 - search("query")  ← Loop!

Debug:
- Add loop detection
- Vary search queries
- Add max retries
```

**2. Hallucinated tool calls**
```
Logs show:
  Agent called: get_stock_price("AAPL")
  Error: Tool 'get_stock_price' not found

Debug:
- Check prompt includes correct tool list
- Validate tool calls before execution
- Fine-tune examples of correct tool use
```

**3. Unexpected costs**
```
Expected: $0.10/task
Actual: $2.50/task

Debug:
- Check token counts (context too large?)
- Count LLM calls (too many refinement loops?)
- Inspect prompts (overly verbose?)
```

**Tools and platforms:**

**LangSmith:**
```python
from langsmith import Client

client = Client()

# Trace agent execution
with client.trace(name="my_agent", tags=["production"]) as run:
    result = agent.run(task)

# View in UI:
- Full trace tree
- Token counts
- Costs
- Latency breakdown
```

**Langfuse:**
```python
from langfuse import Langfuse

langfuse = Langfuse()

# Auto-track
langfuse.observe(agent.run)(task)

# Dashboard shows:
- Cost per user
- Most expensive queries
- Error patterns
- User feedback
```

**Weights & Biases:**
```python
import wandb

wandb.init(project="agent-monitoring")

# Log metrics
wandb.log({
    "success_rate": 0.87,
    "avg_cost": 0.12,
    "latency_p95": 3200
})

# Log examples
wandb.log({"failed_tasks": wandb.Table(dataframe=failed_df)})
```

**Best practices:**

**1. Log everything**
```python
# Every LLM call, tool call, decision point
# Storage is cheap, debugging is expensive
```

**2. Add task_id to all logs**
```python
# Makes it easy to trace full execution
logger.info("step", task_id=task_id, ...)
```

**3. Separate prod/dev environments**
```python
if env == "production":
    agent = ProductionAgent(monitoring=True, strict_validation=True)
else:
    agent = DebugAgent(verbose=True, manual_approval=True)
```

**4. User feedback loop**
```python
# After task completion
result = agent.run(task)

# Ask user
feedback = get_user_feedback()  # thumbs up/down

# Log for analysis
log_feedback(task_id, feedback, result)

# Use to improve
if feedback == "bad":
    add_to_failure_cases(task, result)
```

**5. Automated tests**
```python
# Regression test suite
for test_case in test_suite:
    result = agent.run(test_case.task)

    assert result.success == test_case.expected_success
    assert result.cost < test_case.max_cost
    assert result.latency < test_case.max_latency
```

## Follow-up Questions
- How do you debug non-deterministic agent failures?
- What's a reasonable success rate for production agents?
- How do you attribute costs to different users/teams?
