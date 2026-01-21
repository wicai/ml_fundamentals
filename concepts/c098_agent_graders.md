# Agent Evaluation Graders

**Category:** agents
**Difficulty:** 3
**Tags:** agents, evaluation, graders

## Question
What are the three main types of graders used to evaluate AI agents, and when should you use each?

## What to Cover
- **Set context by**: Explaining that graders are the logic that scores agent performance, and different types have different tradeoffs
- **Must mention**: Three types (code-based, model-based, human), their strengths/weaknesses, when to use each, and how they complement each other
- **Show depth by**: Giving concrete examples of each grader type and discussing calibration between model-based and human graders
- **Avoid**: Only listing grader types without explaining the tradeoffs and when to choose each

## Answer
**Grader**: Logic that scores one or more aspects of agent performance.

**Three types of graders:**

**1. Code-based graders**

Use programmatic logic to evaluate:

```python
# String matching
def grade_exact_match(output, expected):
    return output.strip() == expected.strip()

# Binary test (pass/fail)
def grade_test_suite(agent_code):
    result = subprocess.run(["pytest", "tests/"], capture_output=True)
    return result.returncode == 0

# Static analysis
def grade_code_quality(code):
    lint_errors = run_pylint(code)
    return len(lint_errors) == 0

# Outcome verification
def grade_file_created(output_path, expected_content):
    if not os.path.exists(output_path):
        return False
    return open(output_path).read() == expected_content
```

**Pros:**
- Deterministic and reproducible
- Fast execution
- Objective (no bias)
- Cheap to run at scale

**Cons:**
- Brittle with valid variations (formatting, synonyms)
- Can't handle open-ended tasks
- Requires well-defined expected outputs

**Best for:**
- Coding tasks with test suites
- Tasks with exact expected outputs
- High-volume automated testing

**2. Model-based graders**

Use LLMs with rubrics to evaluate:

```python
def grade_with_llm(task, agent_output, rubric):
    prompt = f"""
    Task: {task}
    Agent output: {agent_output}

    Evaluate the output based on this rubric:
    {rubric}

    Score each dimension 1-5:
    - Correctness: Does it solve the task?
    - Completeness: Are all requirements met?
    - Quality: Is the solution well-structured?

    Return JSON: {{"correctness": X, "completeness": Y, "quality": Z, "reasoning": "..."}}
    """
    return llm.generate(prompt)

# Natural language assertions
def check_constraint(transcript, constraint):
    prompt = f"""
    Transcript: {transcript}

    Did the agent satisfy this constraint?
    Constraint: {constraint}

    Answer YES or NO with brief reasoning.
    """
    return llm.generate(prompt)
```

**Example rubric:**
```
Correctness (1-5):
5: Fully correct, handles edge cases
4: Correct for main case, minor issues
3: Partially correct, some errors
2: Major errors but shows understanding
1: Completely wrong

Helpfulness (1-5):
5: Comprehensive, anticipates needs
4: Addresses request well
3: Basic compliance
2: Partially helpful
1: Unhelpful or harmful
```

**Pros:**
- Handles nuance and open-ended tasks
- Flexible for varied outputs
- Can evaluate subjective quality

**Cons:**
- Non-deterministic (run multiple times)
- Requires human calibration
- Slower and more expensive
- May have blind spots

**Best for:**
- Conversational quality assessment
- Creative/subjective tasks
- Evaluating reasoning quality

**3. Human graders**

Subject matter experts review outputs:

```python
def human_evaluation_workflow(task, agent_output):
    # Create evaluation task
    eval_task = {
        "task": task,
        "output": agent_output,
        "rubric": EVAL_RUBRIC,
        "questions": [
            "Did the agent complete the task correctly?",
            "Was the approach appropriate?",
            "Any safety concerns?",
            "Rate overall quality 1-5"
        ]
    }

    # Send to human reviewers
    responses = []
    for reviewer in get_reviewers(n=3):
        response = reviewer.evaluate(eval_task)
        responses.append(response)

    # Aggregate (majority vote or average)
    return aggregate_responses(responses)
```

**Pros:**
- Gold standard for quality
- Catches issues models miss
- Provides rich qualitative feedback
- Can evaluate safety/alignment

**Cons:**
- Expensive ($5-50+ per evaluation)
- Slow (hours to days)
- Doesn't scale
- Inter-rater variability

**Best for:**
- Validating model-based graders
- Safety-critical applications
- Sampling for quality assurance
- Novel task types

**Combining graders:**

```python
class HybridEvaluator:
    def evaluate(self, task, output):
        results = {}

        # Fast automated checks
        results['tests_pass'] = self.code_grader.run_tests(output)
        results['format_valid'] = self.code_grader.check_format(output)

        # LLM assessment
        results['quality_score'] = self.model_grader.assess_quality(output)
        results['constraint_check'] = self.model_grader.check_constraints(output)

        # Human review for sample
        if random.random() < 0.05:  # 5% sample
            results['human_score'] = self.human_grader.review(output)

        return results
```

**Calibration process:**

```
1. Start with human evaluations (gold standard)
2. Build model-based grader with rubric
3. Compare model vs human scores
4. Adjust rubric until correlation > 0.8
5. Use model-based at scale, human for spot-checks
```

**Domain-specific grader patterns:**

| Domain | Code-based | Model-based | Human |
|--------|-----------|-------------|-------|
| Coding | Test suites, linting | Code review rubric | Architecture review |
| Conversation | State verification | Tone/empathy rubric | User satisfaction |
| Research | Source verification | Groundedness check | Expert fact-check |
| Computer use | UI state checks | Task completion rubric | UX assessment |

## Follow-up Questions
- How do you calibrate model-based graders against human judgment?
- What's the right sampling rate for human evaluation?
- How do you handle disagreement between grader types?
