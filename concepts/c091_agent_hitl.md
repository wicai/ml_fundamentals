# Human-in-the-Loop for Agents

**Category:** agents
**Difficulty:** 3
**Tags:** agents, human_feedback, safety

## Question
What is human-in-the-loop (HITL) for AI agents? When and how do you implement it?

## Answer
**Human-in-the-loop** = Human oversight and intervention in agent execution.

**Why HITL?**
- Agents make mistakes
- High-stakes decisions need human approval
- Build trust gradually
- Learn from human corrections

**HITL patterns:**

**1. Confirmation before action**
```python
class HITLAgent:
    REQUIRES_CONFIRMATION = [
        'send_email',
        'delete_file',
        'make_purchase',
        'modify_database'
    ]

    def execute_action(self, action):
        if action.name in self.REQUIRES_CONFIRMATION:
            # Show human what will happen
            explanation = f"""
            About to: {action.name}
            Arguments: {action.args}
            Impact: {self.explain_impact(action)}
            """

            # Ask for approval
            approved = ask_human(explanation)

            if not approved:
                return "Action cancelled by user"

        return self.tools[action.name](**action.args)
```

**Example interaction:**
```
Agent: I plan to send an email to all customers about the promotion.

Preview:
  To: all_customers@company.com (2,451 recipients)
  Subject: New Year Sale - 50% Off
  Body: [draft email]

Do you approve? [y/n]: y

Agent: ✓ Email sent to 2,451 recipients
```

**2. Review and edit**
```python
class ReviewAgent:
    def run(self, task):
        # Agent creates draft
        draft = self.generate_draft(task)

        # Human reviews and edits
        print(f"Draft:\n{draft}\n")
        print("Options: (a)pprove, (e)dit, (r)egenerate")

        choice = input()

        if choice == 'a':
            return self.finalize(draft)
        elif choice == 'e':
            edited = get_human_edit(draft)
            return self.finalize(edited)
        elif choice == 'r':
            return self.run(task)  # Try again
```

**3. Escalation on uncertainty**
```python
class EscalatingAgent:
    def run(self, task):
        # Try to solve autonomously
        confidence, solution = self.solve(task)

        # Escalate if uncertain
        if confidence < 0.7:
            print(f"I'm {confidence:.0%} confident. Need your help!")
            print(f"My answer: {solution}")

            human_input = input("Your suggestion: ")
            solution = self.refine(solution, human_input)

        return solution
```

**4. Human feedback loop**
```python
class LearningAgent:
    def run(self, task):
        result = self.solve(task)

        # Get human feedback
        feedback = get_feedback(result)  # thumbs up/down + comments

        # Learn from feedback
        if feedback.rating == "bad":
            self.add_negative_example(task, result)
            # Try again with human guidance
            guidance = feedback.comments
            result = self.solve_with_guidance(task, guidance)

        return result
```

**5. Monitoring dashboard**
```
Real-time agent dashboard:

┌─ Current Tasks ──────────────┐
│ Task #123: Sending email     │
│   Status: WAITING_APPROVAL   │
│   [Approve] [Reject] [Edit]  │
│                               │
│ Task #124: Data analysis     │
│   Status: RUNNING             │
│   Progress: 60%               │
└───────────────────────────────┘

┌─ Recent Actions ─────────────┐
│ ✓ Searched web for "ML news" │
│ ✓ Created summary            │
│ ⏸ Needs approval: send email │
└───────────────────────────────┘
```

**When to use HITL:**

**Always require human approval:**
- ❗ Irreversible actions (delete, send email, purchase)
- ❗ High-stakes decisions (medical, legal, financial)
- ❗ Actions affecting others (notifications, messages)

**Escalate to human:**
- ⚠️ Low confidence (< 70%)
- ⚠️ Conflicting information
- ⚠️ Unusual requests
- ⚠️ Edge cases not seen in training

**Optional human review:**
- ℹ️ Draft content (articles, reports)
- ℹ️ Non-critical decisions
- ℹ️ Learning from user preferences

**No human needed:**
- ✅ Safe, reversible actions (search, calculate)
- ✅ High confidence (> 95%)
- ✅ Well-defined tasks

**Implementation approaches:**

**Synchronous (blocking):**
```python
def run_with_approval(task):
    plan = agent.create_plan(task)

    for step in plan:
        if step.requires_approval:
            # Block and wait for human
            approved = wait_for_approval(step)  # Blocking!

            if not approved:
                return "Task cancelled"

        execute(step)
```

**Pros**: Simple, guaranteed approval
**Cons**: Slow, requires human to be online

**Asynchronous (queue):**
```python
class AsyncHITLAgent:
    def run(self, task):
        plan = agent.create_plan(task)

        for step in plan:
            if step.requires_approval:
                # Add to approval queue
                task_id = approval_queue.add(step)

                # Continue with other work
                # Human approves later
                # Agent picks up when approved

            else:
                execute(step)
```

**Pros**: Non-blocking, efficient
**Cons**: More complex, delayed execution

**Tiered approval:**
```python
class TieredApproval:
    def get_approval(self, action):
        risk_level = self.assess_risk(action)

        if risk_level == "low":
            return True  # Auto-approve

        elif risk_level == "medium":
            # Junior reviewer can approve
            return get_approval_from("junior_reviewer")

        elif risk_level == "high":
            # Senior approval required
            return get_approval_from("senior_reviewer")

        elif risk_level == "critical":
            # Multiple approvals needed
            approvals = [
                get_approval_from("senior_reviewer"),
                get_approval_from("security_team")
            ]
            return all(approvals)
```

**UX patterns:**

**1. Suggested actions with one-click approval:**
```
Agent: I found 3 cheap flights to Paris:

1. [$450] Air France, Direct, 8am-10pm [Book]
2. [$420] United, 1 stop, 6am-11pm [Book]
3. [$380] Budget Air, 2 stops, 3am-11pm [Book]

[Cancel] [Let me choose manually]
```

**2. Confidence indicators:**
```
Agent: Here's my analysis (🟢 High confidence)

vs

Agent: Here's my analysis (🟡 Medium confidence - please review)

vs

Agent: Here's my analysis (🔴 Low confidence - needs your input)
```

**3. Explanation of actions:**
```
Agent: I'm about to send an email.

Why: You asked me to notify the team about the meeting
Who: team@company.com (12 people)
What: Meeting tomorrow at 2pm in Conference Room A
Risk: Low (informational only)

[Approve] [Edit message] [Cancel]
```

**Progressive autonomy:**

```python
class ProgressiveAutonomyAgent:
    """Grant more autonomy as agent proves reliable"""

    def __init__(self):
        self.trust_score = 0.0  # Start with no trust

    def execute_action(self, action):
        # Check if action needs approval based on trust
        if self.needs_approval(action):
            approved = get_human_approval(action)

            if approved:
                self.trust_score += 0.1  # Build trust
            else:
                self.trust_score -= 0.2  # Lose trust

            if not approved:
                return "Cancelled"

        result = execute(action)

        # Did human like the result?
        feedback = get_feedback(result)
        if feedback == "good":
            self.trust_score += 0.05

        return result

    def needs_approval(self, action):
        risk = assess_risk(action)

        # Higher trust → less approval needed
        approval_threshold = risk / (1 + self.trust_score)

        return approval_threshold > 0.5
```

**Measuring HITL effectiveness:**

```python
metrics = {
    "approval_rate": 0.92,  # 92% of actions approved
    "edit_rate": 0.15,      # 15% edited before approval
    "avg_approval_time": 45,  # 45 seconds
    "autonomous_rate": 0.60,  # 60% of actions don't need approval
}

# Goal: Increase autonomous_rate while maintaining quality
```

**Real-world examples:**

**GitHub Copilot:**
- Suggests code
- Human reviews and accepts/rejects
- Can edit before accepting

**Grammarly:**
- Suggests edits
- Human decides which to apply
- Learns from user preferences

**Email assistants:**
- Draft emails
- Human reviews before sending
- Critical: never auto-send

**Anthropic Claude:**
- Explains actions before taking them
- Asks for confirmation on sensitive operations
- User can stop at any time

## Follow-up Questions
- How do you balance autonomy and safety with HITL?
- When can you transition from HITL to fully autonomous?
- How do you prevent HITL from becoming a bottleneck?
