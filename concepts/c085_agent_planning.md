# Agent Planning and Reasoning

**Category:** agents
**Difficulty:** 4
**Tags:** agents, planning, reasoning

## Question
How do AI agents plan and break down complex tasks? What are the key planning approaches?

## Answer
**Problem**: LLMs are next-token predictors, not natural planners.
- Greedy, myopic decisions
- Poor at multi-step reasoning
- Struggle with backtracking

**Planning makes agents more capable.**

**Key planning approaches:**

**1. Chain-of-Thought (CoT) planning**
```
User: "Plan a trip to Japan for 2 weeks"

Agent with CoT:
Let me think through this step by step:
1. First, I need to determine the budget
2. Then research flights and best times to visit
3. Identify must-see destinations
4. Create day-by-day itinerary
5. Book accommodations and activities

Now let me start with step 1: What's your budget?
```

**Simple but effective for straightforward tasks.**

**2. ReAct (Reason + Act)**
```
Thought: I need to find population of France
Action: search("France population 2024")
Observation: 67.97 million

Thought: Now I need to compare with Germany
Action: search("Germany population 2024")
Observation: 84.48 million

Thought: I have both numbers, can answer
Answer: Germany has a larger population (84.48M vs 67.97M)
```

**Interleaves thinking and acting.**

**3. Tree-of-Thoughts (ToT)**
```
                    [Initial task]
                   /      |      \
            [Plan A]  [Plan B]  [Plan C]
             /   \       |        /   \
          [A1]  [A2]   [B1]    [C1]  [C2]
           ✗     ✓      ✗       ✓     ✗

Evaluate each path, pursue promising ones.
```

**Implementation:**
```python
class TreeOfThoughts:
    def solve(self, problem, depth=3):
        # Generate multiple possible next steps
        candidates = self.generate_candidates(problem, n=5)

        # Evaluate each candidate
        scored_candidates = []
        for candidate in candidates:
            score = self.evaluate(candidate)
            scored_candidates.append((candidate, score))

        # Keep top K
        top_k = sorted(scored_candidates, key=lambda x: x[1])[:3]

        # Recurse on promising paths
        if depth > 0:
            solutions = []
            for candidate, score in top_k:
                sub_solution = self.solve(candidate, depth-1)
                solutions.append((sub_solution, score))

            return max(solutions, key=lambda x: x[1])

        return top_k[0]
```

**Use case**: Math problems, puzzle solving, creative writing.

**4. Subgoal decomposition**
```python
def plan_task(task):
    # Decompose into subgoals
    subgoals = llm.decompose(task)

    plan = []
    for subgoal in subgoals:
        # Recursively decompose if needed
        if is_complex(subgoal):
            sub_plan = plan_task(subgoal)
            plan.extend(sub_plan)
        else:
            plan.append(subgoal)

    return plan

# Example
task = "Launch a new product"

plan = [
    "1. Market research",
    "  1.1 Identify target audience",
    "  1.2 Analyze competitors",
    "2. Product development",
    "  2.1 Design features",
    "  2.2 Build prototype",
    "3. Marketing campaign",
    ...
]
```

**5. Least-to-Most prompting**
```
Problem: Solve 15 × 23

Decomposition:
Q: What's 15 × 20?
A: 300

Q: What's 15 × 3?
A: 45

Q: What's 300 + 45?
A: 345

Answer: 15 × 23 = 345
```

**Builds complex solutions from simple sub-problems.**

**6. Plan-and-Execute**
```python
class PlanExecuteAgent:
    def run(self, task):
        # 1. Planning phase
        plan = self.create_plan(task)
        print(f"Plan: {plan}")

        # 2. Execution phase
        for step in plan:
            result = self.execute_step(step)

            # 3. Re-plan if needed
            if not result.success:
                print(f"Step {step} failed, re-planning...")
                plan = self.replan(task, completed_steps, failed_step)

        return self.final_result

    def create_plan(self, task):
        prompt = f"""Create a step-by-step plan for: {task}

        Output format:
        1. [First step]
        2. [Second step]
        ...
        """
        return llm(prompt)

    def execute_step(self, step):
        # Actually execute the step
        return execute(step)
```

**7. Hierarchical planning (HRL)**
```
High-level planner:
"To book trip to Japan: 1) Find flights, 2) Book hotels, 3) Plan activities"

Low-level executor for "Find flights":
→ search_flights(destination="Tokyo")
→ filter_by_price()
→ select_best_option()

Low-level executor for "Book hotels":
→ search_hotels(city="Tokyo")
→ check_reviews()
→ make_reservation()
```

**Each level of abstraction has its own agent.**

**8. Iterative refinement**
```python
def solve_with_refinement(task, max_iterations=3):
    solution = initial_attempt(task)

    for i in range(max_iterations):
        # Critique current solution
        critique = llm.critique(solution)

        if critique.is_satisfactory:
            return solution

        # Refine based on critique
        solution = llm.refine(solution, critique)

    return solution
```

**Production examples:**

**OpenAI o1 (reasoning model):**
- Extended chain-of-thought at inference time
- Plans before answering
- ~10-60 seconds of "thinking"
- Better at complex reasoning tasks

**AutoGPT:**
- Creates task list
- Executes tasks sequentially
- Updates plan based on results
- Can get stuck in loops (needs human guidance)

**BabyAGI:**
```python
# Simple task management system
task_queue = [initial_task]

while task_queue:
    # 1. Get next task
    current_task = task_queue.pop(0)

    # 2. Execute it
    result = agent.execute(current_task)

    # 3. Generate new tasks based on result
    new_tasks = agent.create_new_tasks(result, task_queue)

    # 4. Prioritize all tasks
    task_queue = agent.prioritize_tasks(task_queue + new_tasks)
```

**Planning challenges:**

**1. Changing environments**
```
Plan: 1) Search for flights, 2) Book cheapest

Problem: Flight prices changed between steps!
Solution: Re-plan, check preconditions before each step
```

**2. Overplanning**
```
Agent creates 50-step plan for simple task
→ Wastes time planning instead of acting

Solution: Balance planning and execution
```

**3. Hallucinated plans**
```
Agent plans to use tools/APIs that don't exist
→ Fails during execution

Solution: Ground plans in available actions
```

**4. No backtracking**
```
Agent commits to bad plan, can't recover
→ Fails task

Solution: Monitor progress, replan when needed
```

**Best practices:**

**1. Progressive elaboration**
```
Start: High-level plan (3-5 steps)
→ Execute first step
→ Elaborate next step into sub-steps
→ Execute
→ Repeat
```

**2. Checkpoints and validation**
```python
for step in plan:
    result = execute(step)

    # Validate
    if not validate(result):
        # Backtrack or replan
        plan = create_alternative_plan()
```

**3. Mixed planning horizons**
```
Short-term: Next 2-3 steps (detailed)
Medium-term: Next 5-10 steps (outline)
Long-term: End goal (high-level)

Replan short-term frequently, adjust long-term rarely.
```

**Cost-performance tradeoffs:**

```
Simple CoT: Fast, cheap, works for easy tasks
ToT: Slow, expensive, best accuracy
ReAct: Good balance for most tasks
Subgoal decomposition: Medium cost, very interpretable
```

## Follow-up Questions
- When would you use Tree-of-Thoughts vs ReAct?
- How do you validate if a plan is good before executing?
- What causes agents to create circular/infinite plans?
