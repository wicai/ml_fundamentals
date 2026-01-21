# Agent Eval Design Best Practices

**Category:** agents
**Difficulty:** 3
**Tags:** agents, evaluation, design

## Question
What are the best practices for designing effective agent evaluations?

## What to Cover
- **Set context by**: Explaining that good evals are critical for agent development but hard to design well
- **Must mention**: Start early with small sets, design unambiguous tasks with reference solutions, grade outcomes not steps, build isolated environments, read transcripts, and monitor for eval saturation
- **Show depth by**: Discussing common mistakes (ambiguous tasks, step-grading, correlated failures) and how evals fit into the broader development workflow
- **Avoid**: Only listing practices without explaining why they matter and common mistakes to avoid

## Answer
**Why eval design matters:**
- Bad evals give false confidence
- Over-fitted evals stop driving improvement
- Expensive evals slow iteration
- Ambiguous evals produce noise

**Best practices:**

**1. Start early with 20-50 tasks**

Don't wait for hundreds of tasks:

```python
# Start with real user failures
eval_tasks = []

# Mine from production logs
for log in production_logs:
    if log.user_feedback == "unhelpful":
        eval_tasks.append({
            "input": log.user_query,
            "context": log.context,
            "failure_reason": log.feedback_details
        })

# Prioritize by impact
eval_tasks = sorted(eval_tasks,
    key=lambda t: t.get("user_tier", 0) * t.get("frequency", 1),
    reverse=True
)[:50]
```

**Why:**
- Real failures > synthetic tasks
- Small set enables fast iteration
- Can expand as patterns emerge
- Avoids analysis paralysis

**2. Design unambiguous tasks**

Every task needs:
- Clear success criteria
- Reference solution (proves it's solvable)
- Single correct interpretation

```python
# Bad: Ambiguous
{
    "task": "Help the user with their code",
    "success": "User is satisfied"  # How do we know?
}

# Good: Unambiguous
{
    "task": "Fix the IndexError in function parse_date()",
    "input_code": "def parse_date(s): return s.split('-')[3]",
    "test_cases": [
        ("2024-01-15", (2024, 1, 15)),
        ("2023-12-31", (2023, 12, 31))
    ],
    "reference_solution": "def parse_date(s): parts = s.split('-'); return (int(parts[0]), int(parts[1]), int(parts[2]))"
}
```

**3. Create balanced problem sets**

Test both positive and negative cases:

```python
eval_suite = {
    # Should trigger behavior
    "positive_cases": [
        {"task": "Search web for current weather", "expected_tool": "web_search"},
        {"task": "Calculate 15% tip on $50", "expected_tool": "calculator"},
    ],

    # Should NOT trigger behavior
    "negative_cases": [
        {"task": "What's 2+2?", "expected_tool": None},  # Don't need calculator
        {"task": "Tell me about dogs", "expected_tool": None},  # Don't need search
    ]
}

# Prevents overfitting to "always use tool X"
```

**4. Grade outcomes, not steps**

```python
# Bad: Grading exact steps
def grade_steps(transcript, expected_steps):
    for i, step in enumerate(transcript.steps):
        if step != expected_steps[i]:
            return False  # Fails if different approach
    return True

# Good: Grading outcomes
def grade_outcome(final_state, expected_outcome):
    # Don't care HOW agent got there
    return final_state == expected_outcome

# Example: File creation task
def grade_file_task(workspace):
    # Only check final result
    return (
        os.path.exists("output.txt") and
        "Hello World" in open("output.txt").read()
    )
    # Agent could use echo, python, cat, etc. - all valid!
```

**Why:**
- Multiple valid approaches exist
- Penalizing creativity hurts improvement
- Outcome is what users care about

**5. Build isolated environments**

```python
class IsolatedEvalEnvironment:
    def setup(self):
        # Fresh environment per trial
        self.workspace = tempfile.mkdtemp()
        self.container = docker.create_container(
            image="eval-sandbox",
            volumes={self.workspace: "/workspace"}
        )

    def teardown(self):
        # Clean up completely
        self.container.remove()
        shutil.rmtree(self.workspace)

    def run_trial(self, task):
        self.setup()
        try:
            result = self.agent.run(task)
            return self.grade(result)
        finally:
            self.teardown()
```

**Why isolation matters:**
- Prevents state leakage between trials
- Avoids correlated failures (infra issue fails all)
- Reproducible results

**6. Read transcripts regularly**

```python
def manual_transcript_review(sample_size=20):
    """
    Regular transcript review catches grader issues
    """
    transcripts = random.sample(recent_transcripts, sample_size)

    for t in transcripts:
        print(f"Task: {t.task}")
        print(f"Grade: {t.grade}")
        print(f"Transcript:\n{t.full_text}")

        # Human check
        human_grade = input("Your grade (pass/fail): ")

        if human_grade != t.grade:
            log_grader_mismatch(t, human_grade)
            # Grader might be wrong!

# Schedule weekly
schedule.every().monday.do(manual_transcript_review)
```

**Why:**
- Graders can have blind spots
- Catches false positives/negatives
- Reveals new failure modes
- Keeps evals calibrated

**7. Monitor for eval saturation**

```python
def check_saturation(eval_history):
    """
    Saturated eval = no longer useful for improvement
    """
    recent_scores = eval_history[-10:]

    # Check if scores plateaued
    variance = np.var(recent_scores)
    mean = np.mean(recent_scores)

    if mean > 0.95 and variance < 0.01:
        print("WARNING: Eval saturated at high performance")
        print("Consider: Harder tasks, new edge cases")

    if mean < 0.20 and variance < 0.01:
        print("WARNING: Eval saturated at low performance")
        print("Consider: Simpler tasks, decomposed subtasks")

    return {
        "is_saturated": variance < 0.01,
        "ceiling_effect": mean > 0.95,
        "floor_effect": mean < 0.20
    }
```

**Signs of saturation:**
- Pass rate stuck at ~100% (too easy)
- Pass rate stuck at ~0% (too hard)
- No variance across model changes

**Response:**
- Add harder/easier tasks
- Introduce new edge cases
- Split into sub-evaluations

**Common mistakes:**

**1. Synthetic-only tasks**
```python
# Bad: Only synthetic
tasks = [
    "Write a function to add two numbers",
    "Create a hello world program"
]

# Good: Mix real + synthetic
tasks = real_user_failures + edge_cases + synthetic_basics
```

**2. Insufficient trials**
```python
# Bad: 1 trial per task
results = [agent.run(task) for task in tasks]
score = sum(results) / len(results)  # High variance!

# Good: Multiple trials
results = []
for task in tasks:
    trial_results = [agent.run(task) for _ in range(5)]
    results.append(np.mean(trial_results))
score = np.mean(results)  # More reliable
```

**3. Ignoring cost**
```python
# Track cost alongside performance
def evaluate_with_cost(agent, tasks):
    results = []
    for task in tasks:
        start_cost = agent.total_cost
        success = agent.run(task)
        task_cost = agent.total_cost - start_cost

        results.append({
            "success": success,
            "cost": task_cost
        })

    return {
        "pass_rate": np.mean([r["success"] for r in results]),
        "avg_cost": np.mean([r["cost"] for r in results]),
        "cost_per_success": total_cost / num_successes
    }
```

**Evals in the development workflow:**

```
Development cycle:

1. Identify failure mode (production logs)
         ↓
2. Add to eval suite (20-50 tasks)
         ↓
3. Iterate on agent (use evals for fast feedback)
         ↓
4. Deploy improvement
         ↓
5. Monitor production (catch new failures)
         ↓
   (repeat)

Complementary methods:
- Evals: Pre-launch iteration
- A/B tests: Validate user impact
- Monitoring: Post-launch drift detection
- User feedback: Ground truth signal
```

## Follow-up Questions
- How do you decide when to retire or update an eval task?
- What's the right balance between real user failures and synthetic tasks?
- How do you prevent overfitting to your eval set?
