# Prompt Engineering

**Category:** modern_llm
**Difficulty:** 2
**Tags:** prompting, inference, optimization

## Question
What are the key principles and techniques for effective prompt engineering?

## What to Cover
- **Set context by**: Framing prompt engineering as crafting inputs to elicit desired behavior
- **Must mention**: Core principles (be specific, use examples, specify format, give roles), key techniques (CoT, self-consistency, ReAct)
- **Show depth by**: Mentioning gotchas (prompt sensitivity, model-specific differences) and best practices (test multiple phrasings, use delimiters)
- **Avoid**: Just listing techniques without concrete examples or discussing limitations

## Answer
**Prompt Engineering**: Crafting inputs to elicit desired LLM behavior.

**Core principles:**

**1. Be specific and clear**
❌ "Summarize this"
✓ "Summarize this article in 3 bullet points, each under 20 words, focusing on key findings"

**2. Provide examples (few-shot)**
```
Translate English to French:
English: Hello → French: Bonjour
English: Goodbye → French: Au revoir
English: Thank you → French:
```

**3. Break down complex tasks**
```
Instead of: "Analyze this code"

Use:
1. First, identify the main function
2. Then, find potential bugs
3. Finally, suggest improvements
```

**4. Specify format**
```
Output your answer in JSON:
{
  "summary": "...",
  "confidence": 0.0-1.0,
  "sources": [...]
}
```

**5. Give the model a role/persona**
```
You are an expert Python developer with 10 years of experience.
Review this code for best practices.
```

**6. Use delimiters for clarity**
```
Summarize the text between ### markers:

###
[User's text here]
###
```

**Common techniques:**

**Chain-of-Thought:**
```
Let's solve this step by step:
1. First, ...
2. Then, ...
3. Therefore, ...
```

**Self-Consistency:**
- Generate multiple answers
- Take majority vote or most consistent

**ReAct (Reasoning + Acting):**
```
Thought: I need to find population of France
Action: Search "France population 2024"
Observation: 67.97 million
Thought: Now I can answer
Answer: France has a population of approximately 68 million
```

**Tree of Thoughts:**
- Explore multiple reasoning paths
- Backtrack if needed

**Negative prompting:**
```
Do not use technical jargon.
Do not make up information if you don't know.
```

**Gotchas:**

1. **Prompt sensitivity**: Small changes → big differences
2. **Model-specific**: What works for GPT-4 might not for Llama
3. **Diminishing returns**: Long prompts ≠ better results
4. **Cost**: Longer prompts = more tokens = higher cost

**Best practices:**

- Test multiple phrasings
- Use clear delimiters (```, ###, XML tags)
- Iterate based on failure modes
- Keep prompts maintainable (template with variables)

## Follow-up Questions
- How do you systematically optimize prompts?
- What's the trade-off between prompt length and quality?
- Should you include negative examples (what not to do)?
