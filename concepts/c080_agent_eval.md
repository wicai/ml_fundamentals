# Agent Evaluation & Benchmarks

**Category:** agents
**Difficulty:** 3
**Tags:** agents, evaluation, benchmarks

## Question
How do you evaluate AI agents? What are the key benchmarks and metrics?

## Answer
**Challenge**: Agents are harder to evaluate than standard LLMs because:
- Multi-step reasoning
- Tool use correctness
- Task success vs process quality
- Long-running tasks
- Non-deterministic behavior

**Key evaluation dimensions:**

**1. Task success rate**
```
Success rate = # tasks completed correctly / # total tasks
```
- Binary: did it achieve the goal?
- Most important metric

**2. Efficiency**
- Number of steps/tool calls to completion
- Token usage
- Time to completion
- Cost per task

**3. Tool use accuracy**
- Correct tool selection
- Valid arguments
- Error handling

**4. Reasoning quality**
- Logical coherence
- Planning ability
- Adapting to feedback

**Major benchmarks:**

**WebArena** (web navigation)
- Real websites with realistic tasks
- "Book a flight from X to Y under $500"
- Measures: success rate, steps, time
- SOTA: ~35% success (hard!)

**SWE-bench** (software engineering)
- Real GitHub issues from popular repos
- Agent must submit correct PR
- Measures: % issues resolved correctly
- SOTA: ~25% (very hard!)

**GAIA** (general AI assistant)
- Complex multi-step questions
- Requires web search, calculation, reasoning
- Measures: correct answer rate
- SOTA: ~30%

**HotPotQA** (multi-hop reasoning)
- Questions requiring 2+ Wikipedia articles
- Measures: exact match accuracy
- SOTA: ~70% (agents help a lot)

**ToolBench** (tool use)
- 16k+ real-world APIs
- Evaluate tool selection and execution
- Measures: success rate, tool accuracy

**WebShop** (e-commerce)
- Buy specific products matching criteria
- Measures: attribute match score
- SOTA: ~60%

**ALFWorld** (embodied reasoning)
- Text-based household tasks
- "Put a hot apple in the fridge"
- Measures: success rate
- SOTA: ~80%

**Evaluation methodology:**

**1. Automated metrics:**
```python
def evaluate_agent(agent, tasks):
    results = []
    for task in tasks:
        start_time = time.time()

        result = agent.run(task)

        metrics = {
            'success': check_success(result, task.ground_truth),
            'steps': len(agent.trajectory),
            'time': time.time() - start_time,
            'cost': calculate_cost(agent.trajectory),
            'tool_accuracy': evaluate_tools(agent.trajectory)
        }
        results.append(metrics)

    return aggregate_metrics(results)
```

**2. Human evaluation:**
- Task completion quality
- Response appropriateness
- Safety and helpfulness
- User satisfaction

**3. Component testing:**
- Planning module accuracy
- Tool selection accuracy
- Error recovery ability

**Key metrics to track:**

```
Primary:
- Success rate (overall task completion)
- Average cost per task

Secondary:
- Steps to completion
- Tool call accuracy
- Failure mode analysis
- Time to completion

Diagnostic:
- Planning quality score
- Reasoning coherence
- Recovery from errors
```

**Common evaluation challenges:**

**1. Environment differences:**
- Benchmark websites change
- APIs get updated
- Ground truth becomes stale

**2. Partial credit:**
```
Task: "Find and email 3 cheap flights to Paris"
Agent: Finds 2 cheap flights, doesn't email

Success? 0% or 67%?
```

**3. Non-determinism:**
- Same task, different executions
- Need multiple runs for reliability

**4. Cost:**
- Running full benchmarks = $$$
- Each task may take 50+ LLM calls

**Best practices:**

1. **Multi-metric evaluation**: Don't rely on single metric
2. **Error analysis**: Categorize failure modes
3. **Cost tracking**: $/task is critical for production
4. **Synthetic + real tasks**: Mix for coverage
5. **Continuous monitoring**: Track degradation over time

**Example evaluation:**
```
Agent: GPT-4 + ReAct on WebArena

Results:
- Success rate: 34.2%
- Avg steps: 12.3
- Avg cost: $0.45/task
- Tool accuracy: 87%

Failure modes:
- 40%: Infinite loops
- 30%: Wrong tool selection
- 20%: Correct tool, wrong arguments
- 10%: Timeout
```

## Follow-up Questions
- How would you design an eval for a customer support agent?
- What's the difference between agent and LLM evaluation?
- How do you prevent agents from gaming benchmarks?
