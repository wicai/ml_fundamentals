# RL for AI Agents

**Category:** agents
**Difficulty:** 4
**Tags:** agents, reinforcement_learning, training

## Question
How is reinforcement learning used to train AI agents? What are the key approaches?

## Answer
**Why RL for agents?**

Standard LLMs trained on text → poor at:
- Long-term planning
- Tool use
- Sequential decision making
- Learning from interaction

RL helps agents learn from trial and error.

**Key RL approaches for agents:**

**1. RLHF for tool use**
```
1. Collect demonstrations:
   - Humans use tools to complete tasks
   - Record (state, action, tool_call, result)

2. Train reward model:
   - Good: Correct tool, correct args, task progress
   - Bad: Wrong tool, errors, loops

3. RL fine-tuning (PPO):
   - Agent takes actions
   - Get rewards from RM
   - Update policy to maximize reward
```

**2. ReAct + RL**
```
State: Current task, history of actions
Action space: {
  "think": Generate reasoning,
  "act": Call tool with args,
  "finish": Return final answer
}

Reward:
  +10: Task completed correctly
  -1: Each step (encourage efficiency)
  -5: Tool error
  -10: Infinite loop detected
```

**3. Voyager (Minecraft agent)**
- Open-ended RL in Minecraft
- LLM writes code for skills
- Skills stored in library
- RL explores and learns new skills

**Algorithm:**
```
while True:
    # LLM proposes next skill to learn
    skill_code = llm.propose_skill(current_state)

    # Execute in environment
    success, feedback = env.execute(skill_code)

    if success:
        skill_library.add(skill_code)  # Save it!
    else:
        # Use feedback to improve
        skill_code = llm.refine(skill_code, feedback)
```

**4. WebGPT (RL for web search)**
- Agent learns to search web and answer questions
- Action space: search, click, quote, submit answer
- Reward: Human feedback on answer quality

**Training process:**
```
1. Collect trajectories:
   Question → [search, click, click, quote, answer]

2. Reward model:
   - Trained on human preferences
   - Correct answer: +1
   - Incorrect: -1
   - Efficient: bonus

3. PPO update:
   - Sample trajectories from agent
   - Compute advantages
   - Update policy
```

**5. Toolformer (implicit RL)**
- LLM learns when to call tools
- Self-supervised: tools only if they improve prediction

**Algorithm:**
```
1. LLM generates text with potential tool calls
   "The population of France is [search(France population)] 67M"

2. Execute tool, check if helpful:
   perplexity(with tool) < perplexity(without tool) ?

3. Keep tool calls that help, remove others

4. Fine-tune on filtered data
```

**6. Tree-of-Thoughts + RL**
- Agent explores multiple reasoning paths
- RL learns which paths lead to success
- Value network predicts path quality

**Reward design challenges:**

**Dense vs sparse rewards:**
```python
# Sparse (hard to learn)
reward = 1 if task_complete else 0

# Dense (easier, but need to design carefully)
reward = 0
reward += 0.1 * made_progress
reward += 0.5 * correct_tool_use
reward -= 0.05 * num_steps  # Efficiency
reward += 1.0 * task_complete
```

**Reward hacking examples:**
```
Task: Answer user questions

Bad reward: Length of response
→ Agent writes essays for yes/no questions

Bad reward: User satisfaction (no verification)
→ Agent just agrees with user, makes up facts

Good reward: Correct answer + helpfulness (verified)
```

**Key RL algorithms used:**

**PPO (Proximal Policy Optimization):**
- Most popular for agent training
- Stable, sample efficient
- Used in: WebGPT, InstructGPT, agent fine-tuning

**Advantage Actor-Critic (A2C):**
- Value network guides policy
- Useful for multi-step reasoning

**Expert Iteration:**
- Generate trajectories
- Keep successful ones
- Train agent to imitate successful trajectories
- Repeat

**Online vs offline RL:**

**Online RL:**
- Agent interacts with environment
- Learns from own experience
- Expensive (many API calls)

**Offline RL:**
- Train on fixed dataset of trajectories
- Cheaper, more stable
- But limited by dataset quality

**Practical considerations:**

**1. Sample efficiency:**
```
Challenge: Each trajectory = 20 LLM calls = $0.20
1000 episodes = $200
Solution:
- Start with behavior cloning
- Use offline RL
- Reward model instead of real environment
```

**2. Safety:**
```
Problem: RL agent explores, might do dangerous things

Solutions:
- Constrained action space
- Human-in-the-loop for risky actions
- Reward model includes safety
```

**3. Catastrophic forgetting:**
```
Problem: RL fine-tuning → forget general capabilities

Solution:
- Mix RL with SFT data
- KL penalty from original model
- LoRA instead of full fine-tuning
```

**State-of-the-art:**

- **GPT-4 with tool use**: RLHF for function calling
- **Claude with tools**: RL-tuned for appropriate tool use
- **Gemini agents**: RL for multi-modal action taking
- **Adept ACT-1**: RL for computer control

**When to use RL for agents:**

✓ Clear reward signal (task success)
✓ Can simulate environment (cheap exploration)
✓ Need to optimize sequential decisions
✗ Sparse rewards, hard to design
✗ Expensive environment (every call = $$$)
✗ Safety-critical (unpredictable exploration)

## Follow-up Questions
- How do you prevent reward hacking in agent RL?
- What's the difference between RLHF for chat vs agents?
- How does Voyager's skill library work?
