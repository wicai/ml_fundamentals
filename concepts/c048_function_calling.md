# Function Calling / Tool Use

**Category:** modern_llm
**Difficulty:** 3
**Tags:** agents, tools, architecture

## Question
How does function calling work in LLMs and why is it useful?

## Answer
**Function Calling**: LLM outputs structured function calls instead of just text.

**Setup:**
```
1. Define available functions with schemas:

get_weather(location: str, units: str = "celsius") -> dict
search_web(query: str, num_results: int = 5) -> list

2. Include function definitions in system prompt:
"""
You have access to these functions:
- get_weather(location, units): Get weather
- search_web(query, num_results): Search web
...
"""

3. User query: "What's the weather in Paris?"

4. LLM outputs:
{
  "function": "get_weather",
  "arguments": {"location": "Paris", "units": "celsius"}
}

5. Execute function (outside LLM), get result:
{"temp": 15, "conditions": "Cloudy"}

6. Return result to LLM:
"Function result: Temperature is 15°C, cloudy"

7. LLM generates final response:
"The weather in Paris is currently 15°C and cloudy."
```

**Implementation approaches:**

**1. Special tokens (OpenAI API)**
```
<function_call>
{"name": "get_weather", "arguments": {...}}
</function_call>
```

**2. Structured output (fine-tuned)**
Train model to output JSON function calls

**3. Prompt-based (no fine-tuning)**
```
If you need to call a function, output:
FUNCTION: function_name
ARGS: {"arg1": "value1"}
```

**Use cases:**

1. **Calculators**: Accurate math (LLMs bad at arithmetic)
2. **Current information**: Web search, database queries
3. **Actions**: Send email, create calendar event
4. **External APIs**: Weather, stock prices, etc.

**Challenges:**

1. **Hallucinated calls**: Model calls non-existent functions
2. **Argument errors**: Wrong types, missing required args
3. **Infinite loops**: Model keeps calling functions
4. **Security**: User prompt injection to call dangerous functions

**ReAct pattern** (popular framework):
```
Thought: I need current weather
Action: get_weather("Paris")
Observation: 15°C, cloudy
Thought: Now I can answer
Answer: The weather in Paris is...
```

**Modern implementations:**
- **OpenAI Functions**: Built into API
- **LangChain**: Framework for tool use
- **LlamaIndex**: Data connectors with function calling
- **Function-calling fine-tuning**: Gorilla, ToolLLaMA

**Best practices:**
- Validate function outputs before returning to LLM
- Limit number of function calls (prevent loops)
- Sandbox execution (security)
- Clear function descriptions (LLM needs to understand)

## Follow-up Questions
- How do you prevent hallucinated function calls?
- What's the difference between function calling and plugins?
- How do you handle function call failures?
