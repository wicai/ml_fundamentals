# Agent Failure Modes and Recovery

**Category:** agents
**Difficulty:** 3
**Tags:** agents, debugging, reliability

## Question
What are common failure modes for AI agents? How do you design for reliability and recovery?

## Answer
**Common failure modes:**

**1. Infinite loops**
```
Task: "Find cheap flights to Paris"

Agent execution:
1. search_flights("Paris") → No results
2. Think: I should search again
3. search_flights("Paris") → No results (same query!)
4. Think: Let me try searching again...
→ Infinite loop, burns $$
```

**Prevention:**
```python
class LoopDetector:
    def __init__(self, max_same_action=3):
        self.action_history = []
        self.max_same_action = max_same_action

    def check_action(self, action):
        # Count how many times we've done this exact action
        same_count = sum(1 for a in self.action_history[-10:]
                        if a == action)

        if same_count >= self.max_same_action:
            raise InfiniteLoopError(f"Action {action} repeated {same_count} times")

        self.action_history.append(action)
```

**2. Hallucinated tools**
```
Agent: I'll use get_stock_price("AAPL")
System: Tool 'get_stock_price' not found
Agent: Let me try get_stock_data("AAPL")
System: Tool 'get_stock_data' not found
Agent: How about stock_price("AAPL")
→ Keeps hallucinating non-existent tools
```

**Prevention:**
```python
def validate_tool_call(tool_name, args):
    # Check tool exists
    if tool_name not in AVAILABLE_TOOLS:
        error_msg = f"Tool '{tool_name}' doesn't exist."
        error_msg += f"\nAvailable tools: {list(AVAILABLE_TOOLS.keys())}"
        return error_msg

    # Validate arguments
    tool_schema = AVAILABLE_TOOLS[tool_name].schema
    validate_args(args, tool_schema)

    return None  # Valid

# In prompt, include clear tool list
system_prompt += f"\n\nAvailable tools: {list(AVAILABLE_TOOLS.keys())}"
```

**3. Tool errors not handled**
```
Agent: search_web("latest AI news")
Tool: TimeoutError (network down)
Agent: Based on the search results... [hallucinates]
→ Doesn't realize tool failed
```

**Recovery:**
```python
def execute_tool_with_retry(tool_name, args, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = execute_tool(tool_name, args)
            return result

        except ToolError as e:
            if attempt == max_retries - 1:
                # Final attempt failed
                return f"ERROR: Tool {tool_name} failed after {max_retries} attempts: {e}"

            # Exponential backoff
            time.sleep(2 ** attempt)

    # Feed error back to agent so it can try alternative approach
```

**4. Context overflow**
```
Agent has long conversation
Token count: 8000/8000 (context full!)
Next message: Truncated, loses important context
Agent: Confused, repeats questions
```

**Prevention:**
```python
class ContextManager:
    def __init__(self, max_tokens=8000):
        self.messages = []
        self.max_tokens = max_tokens

    def add_message(self, msg):
        self.messages.append(msg)

        # Check if over limit
        if self.count_tokens() > self.max_tokens:
            # Summarize old messages
            old_msgs = self.messages[:len(self.messages)//2]
            summary = llm.summarize(old_msgs)

            # Replace with summary
            self.messages = [
                {"role": "system", "content": f"Previous context: {summary}"}
            ] + self.messages[len(self.messages)//2:]
```

**5. Irreversible mistakes**
```
Agent: Deleting old files...
Agent: delete_file("project.zip")
System: ✓ Deleted
User: WAIT that was my project backup!
→ Too late, file gone
```

**Prevention:**
```python
DANGEROUS_OPERATIONS = ['delete', 'send_email', 'purchase']

def execute_action(action):
    if action.name in DANGEROUS_OPERATIONS:
        # Require explicit confirmation
        print(f"About to: {action}")
        print(f"This action is IRREVERSIBLE")

        confirm = input("Type 'confirm' to proceed: ")

        if confirm != 'confirm':
            return "Action cancelled"

        # Log for audit trail
        audit_log.append({
            'action': action,
            'timestamp': now(),
            'user_confirmed': True
        })

    return execute(action)
```

**6. Cost runaway**
```
Agent stuck in loop
1000 LLM calls × $0.10 = $100
Catches fire 🔥
```

**Circuit breaker:**
```python
class CostCircuitBreaker:
    def __init__(self, max_cost=5.0, max_calls=50):
        self.total_cost = 0
        self.call_count = 0
        self.max_cost = max_cost
        self.max_calls = max_calls

    def check_limit(self, estimated_cost):
        if self.total_cost + estimated_cost > self.max_cost:
            raise CostLimitExceeded(
                f"Would exceed cost limit: ${self.total_cost + estimated_cost:.2f} > ${self.max_cost}"
            )

        if self.call_count >= self.max_calls:
            raise CallLimitExceeded(
                f"Exceeded max calls: {self.call_count} >= {self.max_calls}"
            )

    def record_call(self, cost):
        self.total_cost += cost
        self.call_count += 1
```

**7. Wrong tool selection**
```
User: "Calculate 15% of $200"
Agent: search_web("15% of 200")  ← Wrong tool!
→ Should use calculator, not search
```

**Improvement:**
```python
# Better tool descriptions
tools = {
    "calculator": {
        "description": "Evaluate mathematical expressions. Use for ANY calculation, arithmetic, percentages, etc.",
        "examples": ["15% of 200", "sqrt(144)", "compound interest"]
    },
    "search": {
        "description": "Search web for CURRENT information not in your training. Do NOT use for calculations.",
        "examples": ["latest news", "weather", "stock prices"]
    }
}
```

**8. Silent failures**
```
Agent: Generating report...
[Error occurs internally]
Agent: Here's the report! [Returns empty/wrong result]
User: This report is blank?!
```

**Better error handling:**
```python
def run_task(task):
    try:
        result = agent.execute(task)

        # Validate result
        if not is_valid_result(result):
            raise InvalidResultError("Result failed validation")

        return result

    except Exception as e:
        # Don't hide errors from user
        error_msg = f"I encountered an error: {type(e).__name__}: {str(e)}"
        error_msg += "\nWould you like me to try a different approach?"

        return error_msg
```

**Recovery strategies:**

**1. Retry with backoff**
```python
def resilient_execute(action, max_retries=3):
    for i in range(max_retries):
        try:
            return execute(action)
        except TransientError:
            if i == max_retries - 1:
                raise
            time.sleep(2 ** i)  # 1s, 2s, 4s
```

**2. Fallback plans**
```python
def execute_with_fallback(primary_plan):
    try:
        return execute(primary_plan)
    except ToolError as e:
        # Try alternative approach
        fallback_plan = create_fallback_plan(primary_plan, error=e)
        return execute(fallback_plan)
```

**3. Graceful degradation**
```python
def get_weather(location):
    try:
        # Try primary API
        return weather_api.get(location)
    except APIError:
        try:
            # Fallback to alternative API
            return backup_weather_api.get(location)
        except APIError:
            # Degrade to approximate answer
            return "I couldn't get real-time weather, but historically {location} is {typical_weather}"
```

**4. Checkpointing**
```python
class CheckpointedAgent:
    def run(self, task):
        checkpoints = []

        for step in self.plan(task):
            # Save state before each step
            checkpoint = self.save_state()
            checkpoints.append(checkpoint)

            try:
                result = self.execute(step)
            except Exception as e:
                # Restore to last checkpoint
                self.restore_state(checkpoints[-1])

                # Try alternative
                alternative = self.replan(step, error=e)
                result = self.execute(alternative)

        return result
```

**Reliability best practices:**

**1. Fail fast, fail loud**
```python
# Bad: Silently continue with wrong state
if result is None:
    result = ""  # Pretend everything is fine

# Good: Fail immediately with clear error
if result is None:
    raise ValueError("Expected result but got None at step X")
```

**2. Idempotency**
```python
# Make operations safe to retry
def send_notification(user_id, message, notification_id):
    # Check if already sent
    if notification_sent(notification_id):
        return "Already sent"

    # Send
    send(user_id, message)
    mark_sent(notification_id)

# Can safely retry without duplicate sends
```

**3. Comprehensive logging**
```python
# Log everything for debugging
logger.info("step_start", step=step, state=state)
try:
    result = execute(step)
    logger.info("step_success", result=result)
except Exception as e:
    logger.error("step_failed", error=str(e), traceback=...)
    raise
```

**4. Health checks**
```python
def health_check():
    # Verify agent is functioning
    checks = {
        "llm_available": check_llm(),
        "tools_available": check_tools(),
        "within_budget": check_budget(),
    }

    if not all(checks.values()):
        alert(f"Health check failed: {checks}")
```

**Measuring reliability:**

```python
reliability_metrics = {
    "success_rate": 0.87,      # 87% of tasks succeed
    "retry_rate": 0.15,        # 15% need retries
    "error_rate": 0.13,        # 13% hit errors
    "recovery_rate": 0.60,     # 60% of errors recovered
    "mtbf": 1000,              # Mean time between failures (tasks)
}
```

## Follow-up Questions
- How do you test agent reliability before deploying?
- What's an acceptable error rate for production agents?
- How do you prevent infinite loops without being too restrictive?
