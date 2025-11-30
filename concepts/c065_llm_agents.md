# LLM Agents

**Category:** modern_llm
**Difficulty:** 3
**Tags:** agents, tools, applications

## Question
What are LLM agents and how do they differ from standard LLM applications?

## Answer
**LLM Agent**: LLM that can take actions, use tools, and iterate toward a goal.

**Standard LLM:**
```
Input: Prompt
Output: Text
Done.
```

**LLM Agent:**
```
1. Receive goal/task
2. Reason about next action
3. Take action (call tool, query database, etc.)
4. Observe result
5. Repeat until goal achieved
```

**Key components:**

**1. Reasoning loop:**
```
while not done:
  thought = LLM("Current state: [state]. What should I do next?")
  action = parse_action(thought)
  observation = execute_action(action)
  state = update_state(observation)
```

**2. Tools/Actions:**
- Web search
- Calculator
- Code execution
- Database queries
- API calls

**3. Memory:**
- Short-term: Current conversation/state
- Long-term: Vector database of past interactions

**4. Planning:**
- Decompose task into sub-tasks
- Multi-step reasoning

**Frameworks:**

**ReAct (Reason + Act):**
```
Thought: I need to find the population of France
Action: search("France population 2024")
Observation: 67.97 million
Thought: Now I can calculate the density
Action: calculator("67.97 / 643.801")  # Area of France
Observation: 105.6
Thought: I have the answer
Answer: France has a population density of ~106 people/km²
```

**AutoGPT:**
- Autonomous agent with goals
- Self-prompts to make progress
- Can get stuck in loops

**BabyAGI:**
- Task management system
- Creates, prioritizes, executes tasks
- Simple but effective

**LangChain Agents:**
- Framework for building agents
- Predefined agent types (zero-shot, conversational, etc.)
- Tool ecosystem

**Challenges:**

**1. Reliability:**
```
Agent decides: search("France population")
Error: Network timeout
Agent confused, doesn't retry
Failure.
```

**2. Cost:**
- Many LLM calls per task
- $0.01/call × 50 calls = $0.50 per task

**3. Hallucinated actions:**
- Calls non-existent tools
- Wrong arguments to tools

**4. Infinite loops:**
```
Thought: I should search for X
Action: search(X)
Observation: No results
Thought: I should search for X
Action: search(X)
...
```

**5. Security:**
- Agent can call dangerous functions
- Prompt injection attacks

**Best practices:**

1. **Limited action space**: Only safe, necessary tools
2. **Timeouts**: Max iterations, max cost
3. **Human-in-the-loop**: Confirm before destructive actions
4. **Logging**: Track all actions for debugging
5. **Validation**: Check action arguments before execution

**Examples:**

**Customer support agent:**
```
Tools: Search docs, query database, create ticket
Goal: Resolve customer issue

1. Search docs for similar issue
2. If found, provide solution
3. If not, query database for user info
4. Create ticket and notify human
```

**Data analysis agent:**
```
Tools: SQL query, Python execution, plot
Goal: Answer business question

1. Query database
2. Analyze with Python/pandas
3. Generate visualization
4. Summarize findings
```

**Personal assistant:**
```
Tools: Calendar, email, web search, todo list
Goal: Manage user's schedule

1. Check calendar for conflicts
2. Search for restaurant recommendations
3. Book reservation
4. Add to calendar
5. Send confirmation email
```

**When to use agents:**

✓ Multi-step tasks requiring tools
✓ Tasks with clear success criteria
✓ Tolerate some failures/retries
✗ Simple single-step tasks
✗ Need 100% reliability
✗ Cost-sensitive

## Follow-up Questions
- How do you prevent infinite loops in agents?
- What's the difference between ReAct and chain-of-thought?
- How do you make agents more reliable?
