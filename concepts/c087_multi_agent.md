# Multi-Agent Systems

**Category:** agents
**Difficulty:** 4
**Tags:** agents, multi_agent, collaboration

## Question
What are multi-agent systems? When and how do you use multiple agents together?

## Answer
**Multi-agent system** = Multiple AI agents working together to solve tasks.

**Why multiple agents?**
1. **Specialization**: Each agent is expert in one domain
2. **Parallel work**: Multiple agents work simultaneously
3. **Debate/verification**: Agents check each other's work
4. **Modularity**: Easier to build and maintain

**Architecture patterns:**

**1. Sequential pipeline**
```
User request
    ↓
[Planner Agent] → Creates plan
    ↓
[Research Agent] → Gathers information
    ↓
[Analysis Agent] → Analyzes data
    ↓
[Writer Agent] → Writes report
    ↓
Final output
```

**Example: Code generation system**
```python
class CodeGenPipeline:
    def __init__(self):
        self.spec_agent = SpecificationAgent()
        self.code_agent = CodeAgent()
        self.test_agent = TestAgent()
        self.review_agent = ReviewAgent()

    def generate(self, requirements):
        # 1. Create spec
        spec = self.spec_agent.create_spec(requirements)

        # 2. Write code
        code = self.code_agent.write_code(spec)

        # 3. Generate tests
        tests = self.test_agent.create_tests(spec, code)

        # 4. Review
        review = self.review_agent.review(code, tests)

        if review.approved:
            return code
        else:
            # Fix issues and retry
            code = self.code_agent.fix_issues(code, review.feedback)
            return code
```

**2. Debate/multi-perspective**
```
Question
    ↓
[Agent A] → Proposes answer
[Agent B] → Proposes alternative
[Agent C] → Proposes another view
    ↓
[Judge Agent] → Evaluates, picks best
    ↓
Final answer
```

**Example: Math problem solving**
```python
class DebateSystem:
    def solve(self, problem):
        # Multiple agents propose solutions
        solutions = []
        for agent in self.agents:
            solution = agent.solve(problem)
            solutions.append(solution)

        # Agents critique each other
        for solution in solutions:
            critiques = []
            for agent in self.agents:
                critique = agent.critique(solution)
                critiques.append(critique)

            solution['critiques'] = critiques

        # Judge selects best
        best = self.judge.select_best(solutions)
        return best
```

**3. Coordinator-worker**
```
        [Coordinator]
        /    |    \
       /     |     \
[Worker1] [Worker2] [Worker3]
```

**Example: Research agent**
```python
class ResearchCoordinator:
    def research(self, topic):
        # Coordinator plans research
        subtopics = self.plan_research(topic)

        # Assign subtopics to workers
        results = []
        for subtopic in subtopics:
            worker = self.assign_worker(subtopic)
            result = worker.research(subtopic)
            results.append(result)

        # Coordinator synthesizes
        final_report = self.synthesize(results)
        return final_report
```

**4. Collaborative agents (shared context)**
```
        Shared Memory
           /    \
          /      \
    [Agent A] [Agent B]
```

**Example: Coding assistants**
```python
class SharedContext:
    def __init__(self):
        self.codebase = {}
        self.plan = None

# Both agents access same context
frontend_agent = FrontendAgent(context)
backend_agent = BackendAgent(context)

# Frontend agent updates context
frontend_agent.create_ui()  # Updates context.codebase

# Backend agent uses updated context
backend_agent.create_api()  # Sees frontend changes
```

**5. Adversarial agents**
```
[Red Team Agent] ← vs → [Blue Team Agent]
```

**Example: Security testing**
```python
class SecurityTest:
    def __init__(self):
        self.attacker = RedTeamAgent()  # Tries to break system
        self.defender = BlueTeamAgent()  # Improves security

    def iterate(self, system):
        for round in range(10):
            # Attacker finds vulnerability
            attack = self.attacker.find_vulnerability(system)

            # Defender patches
            if attack.successful:
                patch = self.defender.create_patch(attack)
                system.apply_patch(patch)

        return system  # More secure after iterations
```

**Communication protocols:**

**1. Message passing**
```python
class Agent:
    def __init__(self, mailbox):
        self.mailbox = mailbox

    def send_message(self, to_agent, message):
        self.mailbox.send(to_agent, message)

    def receive_messages(self):
        return self.mailbox.receive(self.id)

# Usage
agent_a.send_message("agent_b", {
    "type": "request",
    "content": "Please analyze this data"
})
```

**2. Shared memory/blackboard**
```python
class Blackboard:
    def __init__(self):
        self.state = {}

    def write(self, key, value):
        self.state[key] = value

    def read(self, key):
        return self.state.get(key)

# Agents communicate via blackboard
agent_a.write_to_blackboard("research_results", data)
agent_b.read_from_blackboard("research_results")  # Gets data
```

**3. Hierarchical messaging**
```python
coordinator.broadcast("Start task X")
  → All workers receive message
  → Workers report results back to coordinator
```

**Real-world examples:**

**MetaGPT (software company simulation)**
```
CEO Agent: Sets product direction
Product Manager: Creates requirements
Architect: Designs system
Engineer: Writes code
QA: Tests code

Communication: Sequential handoffs
Result: Working software from prompt
```

**ChatDev (similar to MetaGPT)**
```
Roles: Designer, Programmer, Tester, Reviewer
Phases: Design → Coding → Testing → Documentation
Each agent contributes in their phase
```

**AutoGen (Microsoft)**
```python
# Define agents
assistant = AssistantAgent(name="assistant")
user_proxy = UserProxyAgent(name="user")
critic = CriticAgent(name="critic")

# Conversation
user_proxy.initiate_chat(
    assistant,
    message="Write Python code to analyze stock data"
)

# Assistant writes code
# Critic reviews code
# Assistant improves based on feedback
# Iterate until critic approves
```

**CAMEL (role-playing framework)**
```python
# Two agents with different roles
ai_user = CAMELAgent(role="User", task="Design a game")
ai_assistant = CAMELAgent(role="Assistant", task="Implement game")

# They chat and collaborate
for turn in range(10):
    user_msg = ai_user.step(ai_assistant_msg)
    assistant_msg = ai_assistant.step(user_msg)
```

**Challenges:**

**1. Coordination overhead**
```
Single agent: 5 LLM calls, $0.05
Multi-agent: 20 LLM calls (coordination), $0.20

Only worth it if multi-agent is 4x better!
```

**2. Communication bottlenecks**
```
All agents wait for coordinator → slow
Solution: Parallel work, async communication
```

**3. Conflicting outputs**
```
Agent A: "Solution is X"
Agent B: "Solution is Y"

Need: Conflict resolution mechanism
```

**4. State synchronization**
```
Agent A updates shared state
Agent B reads stale state → incorrect decision

Solution: Versioning, locks, or event-driven updates
```

**When to use multi-agent:**

✓ **Clearly separable subtasks** (research + write + review)
✓ **Need multiple perspectives** (debate improves accuracy)
✓ **Specialized expertise** (legal + technical + business)
✓ **Parallel work possible** (multiple independent analyses)

✗ **Simple tasks** (one agent sufficient)
✗ **Tight coupling** (agents need constant communication)
✗ **Cost-sensitive** (multiple agents = higher cost)

**Design principles:**

1. **Clear roles**: Each agent has specific responsibility
2. **Minimal communication**: Reduce coordination overhead
3. **Modularity**: Agents can be swapped/upgraded
4. **Graceful degradation**: System works if one agent fails

**Example architecture:**

```python
class MultiAgentSystem:
    def __init__(self):
        self.planner = PlannerAgent()
        self.workers = [
            ResearchAgent(),
            AnalysisAgent(),
            WriterAgent()
        ]
        self.reviewer = ReviewAgent()

    def execute_task(self, task):
        # 1. Plan
        plan = self.planner.create_plan(task)

        # 2. Execute in parallel
        results = []
        for subtask in plan.subtasks:
            worker = self.assign_worker(subtask)
            result = worker.execute(subtask)
            results.append(result)

        # 3. Review
        draft = self.combine_results(results)
        final = self.reviewer.review(draft)

        return final
```

## Follow-up Questions
- How do you debug multi-agent systems?
- What's the cost/benefit tradeoff of multi-agent vs single agent?
- How do you prevent infinite conversation loops between agents?
