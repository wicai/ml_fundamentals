# Tool Use and Function Calling

**Category:** agents
**Difficulty:** 3
**Tags:** agents, tools, function_calling

## Question
How do agents use tools and function calling? What are best practices?

## Answer
**Tool use = Agents can call external functions/APIs**

**Example:**
```
User: What's the weather in SF?

Without tools:
LLM: I don't have access to real-time weather data.

With tools:
LLM → calls get_weather("San Francisco")
API → returns {temp: 62, conditions: "cloudy"}
LLM: It's 62°F and cloudy in San Francisco.
```

**How it works:**

**1. Tool definition:**
```python
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'San Francisco'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    }
]
```

**2. System prompt includes tools:**
```
You have access to the following tools:
- get_weather(location, unit): Get weather
- search_web(query): Search the web
- calculator(expression): Evaluate math

To use a tool, output:
<tool_call>
  <name>get_weather</name>
  <args>{"location": "SF", "unit": "fahrenheit"}</args>
</tool_call>
```

**3. LLM outputs tool call:**
```
User: What's 15% tip on $42.50?

LLM: <tool_call>
  <name>calculator</name>
  <args>{"expression": "42.50 * 0.15"}</args>
</tool_call>
```

**4. System executes tool:**
```python
tool_result = execute_tool(
    name="calculator",
    args={"expression": "42.50 * 0.15"}
)
# Returns: 6.375
```

**5. Result fed back to LLM:**
```
Tool result: 6.375

LLM: A 15% tip on $42.50 would be $6.38 (rounded).
```

**Modern approaches:**

**OpenAI Function Calling:**
```python
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's weather in Tokyo?"}],
    functions=[{
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": { ... }
    }],
    function_call="auto"  # Let model decide
)

if response.get("function_call"):
    # Model wants to call a function
    function_name = response["function_call"]["name"]
    args = json.loads(response["function_call"]["arguments"])

    # Execute
    result = execute_function(function_name, args)

    # Send result back
    response = openai.ChatCompletion.create(
        messages=[
            ...,
            {"role": "function", "name": function_name, "content": str(result)}
        ]
    )
```

**Anthropic Tool Use:**
```python
response = anthropic.messages.create(
    model="claude-3-opus",
    messages=[{"role": "user", "content": "Weather in Tokyo?"}],
    tools=[{
        "name": "get_weather",
        "description": "Get weather",
        "input_schema": { ... }
    }]
)

if response.stop_reason == "tool_use":
    tool_use = response.content[0]  # Extract tool call
    result = execute_tool(tool_use.name, tool_use.input)

    # Continue conversation with result
    ...
```

**Best practices:**

**1. Clear tool descriptions:**
```python
# Bad
"description": "Gets weather"

# Good
"description": "Get current weather conditions and temperature for a specific city. Use this when the user asks about weather, temperature, or conditions. Returns temperature, conditions, humidity, and wind speed."
```

**2. Validate tool arguments:**
```python
def execute_tool(name, args):
    # Validate before executing
    schema = tools[name]["parameters"]
    validate(args, schema)  # Check types, required fields

    # Sanitize
    if name == "search_web":
        args["query"] = sanitize_query(args["query"])

    # Execute
    return TOOL_FUNCTIONS[name](**args)
```

**3. Error handling:**
```python
try:
    result = execute_tool(name, args)
except ToolError as e:
    # Feed error back to LLM
    result = f"Error: {str(e)}. Please try again with different arguments."
```

**4. Tool chaining:**
```python
# Agent can call multiple tools in sequence
User: "Find the cheapest flight to Paris next week and add it to my calendar"

Agent:
1. search_flights(destination="Paris", date="next week")
   → Returns flight options
2. sort_by_price(flights)
   → Returns cheapest
3. add_to_calendar(event=flight_details)
   → Confirms
```

**5. Parallel tool calling:**
```python
User: "What's the weather in Tokyo, London, and NYC?"

# Call all three in parallel
tools_to_call = [
    ("get_weather", {"location": "Tokyo"}),
    ("get_weather", {"location": "London"}),
    ("get_weather", {"location": "NYC"})
]

results = parallel_execute(tools_to_call)
```

**Common pitfalls:**

**1. Hallucinated tool calls:**
```
Agent: <tool_call>
  <name>get_stock_price</name>  ← Tool doesn't exist!
  ...
</tool_call>

Solution: Return error, let agent retry
```

**2. Wrong arguments:**
```
Agent: get_weather(location=123)  ← Should be string!

Solution: Validate and return clear error
```

**3. Infinite loops:**
```
Agent: Let me search for that.
  → search_web("query")
  → No results found
Agent: Let me search for that.
  → search_web("query")  ← Same query again!
  ...

Solution: Track history, limit retries, vary approach
```

**4. Security issues:**
```python
# DANGEROUS
def execute_code(code):
    exec(code)  # Never do this!

# SAFE
def calculator(expression):
    # Parse and validate
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        raise ValueError("Invalid characters")

    # Safe eval
    return safe_eval(expression)
```

**Advanced patterns:**

**Tool selection with examples:**
```
When user asks about weather → use get_weather
When user asks about math → use calculator
When user asks to remember something → use save_note

Examples:
Q: "What's 50 * 30?"
A: <tool_call>calculator("50 * 30")</tool_call>

Now you try:
Q: "What's the temperature in Boston?"
```

**Conditional tool availability:**
```python
# Only show tools relevant to current context
if user_is_authenticated:
    tools.append(send_email_tool)

if user_location_available:
    tools.append(nearby_restaurants_tool)
```

**Tool composition:**
```python
# High-level tool built from low-level tools
def book_restaurant(name, time, party_size):
    # Uses multiple tools internally
    restaurant = search_restaurant(name)
    availability = check_availability(restaurant.id, time)
    if availability:
        confirmation = make_reservation(restaurant.id, time, party_size)
        send_email(user.email, confirmation)
    return confirmation
```

## Follow-up Questions
- How do you prevent agents from calling dangerous tools?
- What's the difference between function calling and tool use?
- How would you implement parallel tool execution?
