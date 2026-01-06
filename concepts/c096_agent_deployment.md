# Production Agent Deployment

**Category:** agents
**Difficulty:** 3
**Tags:** agents, deployment, production

## Question
How do you deploy AI agents to production? What are the key considerations?

## Answer
**Production deployment considerations:**

**1. Scalability**
```python
# Handle load spikes
- Auto-scaling based on queue depth
- Rate limiting per user
- Async task processing
- Connection pooling to LLM APIs

# Architecture
┌─ Load Balancer ─┐
│                  │
├─ Agent Server 1 ─┤
├─ Agent Server 2 ─┤  → LLM API (OpenAI/Anthropic)
├─ Agent Server 3 ─┤  → Vector DB (Pinecone)
│                  │  → Redis (state)
└──────────────────┘
```

**2. Cost management**
```python
class ProductionAgent:
    def __init__(self):
        self.cost_tracker = CostTracker(
            budget_per_user_daily=10.0,
            budget_per_task=1.0
        )

    def run(self, task, user_id):
        # Check budget
        if self.cost_tracker.would_exceed_budget(user_id):
            return "Daily budget exceeded. Try again tomorrow."

        # Track costs
        with self.cost_tracker.track(user_id):
            result = self.execute(task)

        return result
```

**3. Latency optimization**
```python
# Target latencies
targets = {
    "planning": "< 2s",
    "tool_execution": "< 1s",
    "response_generation": "< 3s",
    "total": "< 10s"
}

# Optimization strategies
- Use streaming for perceived latency
- Cache common tool results
- Parallel tool execution
- Haiku for simple tasks, Sonnet for complex
```

**4. Monitoring**
```python
# Key metrics
metrics = {
    "requests_per_second": 100,
    "success_rate": 0.87,
    "p50_latency": 3200,
    "p95_latency": 8500,
    "error_rate": 0.13,
    "cost_per_task": 0.15,
    "daily_cost": 1500.00
}

# Alerts
if metrics["success_rate"] < 0.80:
    alert("Success rate degraded!")

if metrics["cost_per_task"] > 0.50:
    alert("Cost spike detected!")
```

**5. Security**
```python
# Input validation
def validate_input(user_input):
    # Rate limiting
    if exceeded_rate_limit(user_id):
        raise RateLimitError()

    # Input sanitization
    if contains_injection(user_input):
        raise SecurityError("Potential injection detected")

    # Size limits
    if len(user_input) > MAX_INPUT_SIZE:
        raise ValueError("Input too large")

# Output filtering
def safe_output(response):
    # Remove PII
    response = redact_pii(response)

    # Remove credentials
    response = redact_credentials(response)

    return response
```

**Deployment patterns:**

**1. Serverless (AWS Lambda, Cloud Run)**
```yaml
# Pros: Auto-scaling, pay-per-use
# Cons: Cold starts, timeout limits

service: agent-api
provider:
  name: aws
  runtime: python3.11
  timeout: 300  # 5 minutes max

functions:
  agent:
    handler: agent.run
    events:
      - http:
          path: /agent
          method: post
    environment:
      OPENAI_API_KEY: ${env:OPENAI_API_KEY}
```

**2. Container-based (Kubernetes)**
```yaml
# Pros: Full control, long-running tasks
# Cons: More complex, manage scaling

apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent
  template:
    spec:
      containers:
      - name: agent
        image: mycompany/agent:v1.2.3
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: anthropic
```

**3. Managed services**
```python
# OpenAI Assistants API - fully managed
assistant = client.beta.assistants.create(
    model="gpt-4-turbo",
    tools=[{"type": "code_interpreter"}]
)

# No infrastructure to manage
# Pay per API call
```

**CI/CD pipeline:**

```yaml
# .github/workflows/deploy.yml
name: Deploy Agent

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Run tests
        run: |
          pytest tests/
          pytest tests/integration/ --slow

      - name: Run safety eval
        run: |
          python eval/safety_tests.py

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: |
          kubectl apply -f k8s/staging/

      - name: Smoke test
        run: |
          python scripts/smoke_test.py --env=staging

      - name: Deploy to production (canary)
        run: |
          # 10% traffic to new version
          kubectl apply -f k8s/prod/canary.yaml

      - name: Monitor metrics
        run: |
          python scripts/monitor_canary.py --duration=30m

      - name: Full rollout
        if: success()
        run: |
          kubectl apply -f k8s/prod/
```

**Canary deployment:**

```python
# Route 10% of traffic to new version
traffic_split = {
    "v1.2.3": 0.10,  # New version
    "v1.2.2": 0.90   # Old version
}

# Monitor for issues
canary_metrics = monitor_for(duration="30m")

if canary_metrics["error_rate"] > 0.15:
    rollback("v1.2.2")
else:
    # Gradually increase traffic
    traffic_split["v1.2.3"] = 0.50  # 50%
    # ... eventually 100%
```

**Error handling:**

```python
class ResilientAgent:
    def run(self, task):
        try:
            return self.execute(task)

        except OpenAIError as e:
            # Retry with backoff
            return self.retry_with_backoff(task)

        except ToolError as e:
            # Try fallback approach
            return self.execute_with_fallback(task)

        except Exception as e:
            # Log for debugging
            logger.error("Unexpected error", exc_info=e)

            # Return safe error to user
            return "I encountered an error. Please try again."
```

**Feature flags:**

```python
class FeatureFlaggedAgent:
    def run(self, task):
        # Check feature flags
        if feature_enabled("use_gpt4_turbo"):
            model = "gpt-4-turbo"
        else:
            model = "gpt-4"

        if feature_enabled("parallel_tool_execution"):
            return self.run_with_parallel_tools(task, model)
        else:
            return self.run_sequential(task, model)

# Can enable/disable features without deployment
```

**Data retention:**

```python
# GDPR compliance
class GDPRCompliantAgent:
    def process(self, user_input):
        # Log with retention policy
        self.log(
            user_input,
            retention_days=30,  # Auto-delete after 30 days
            pii_redacted=True
        )

    def handle_deletion_request(self, user_id):
        # Delete all user data
        db.delete_where(user_id=user_id)
        logs.delete_where(user_id=user_id)
        vector_db.delete_where(user_id=user_id)
```

## Follow-up Questions
- How do you handle database migrations for agent state?
- What's your rollback strategy if a deployment goes wrong?
- How do you test agents before production?
