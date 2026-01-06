# Agent Frameworks and Tools

**Category:** agents
**Difficulty:** 3
**Tags:** agents, frameworks, tools

## Question
What are the major AI agent frameworks? How do they compare?

## Answer
**Agent frameworks** = Libraries and tools for building AI agents faster.

**Major frameworks:**

**1. LangChain**

**What it is**: Most popular agent framework (Python/JS)

**Key features:**
```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

# Define tools
tools = [
    Tool(
        name="Search",
        func=search_web,
        description="Search the web for current information"
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for math"
    )
]

# Create agent
agent = initialize_agent(
    tools=tools,
    llm=OpenAI(temperature=0),
    agent="zero-shot-react-description",  # Agent type
    verbose=True
)

# Run
result = agent.run("What's 15% tip on $45 flight to Paris price?")
```

**Pros:**
- ✓ Huge ecosystem (100+ integrations)
- ✓ Many agent types built-in
- ✓ Active community
- ✓ Good documentation

**Cons:**
- ✗ Abstraction overload (complex API)
- ✗ Frequent breaking changes
- ✗ Can be slow (many unnecessary calls)

**Best for**: Prototyping, experimentation

**2. AutoGPT**

**What it is**: Autonomous agent that creates its own tasks

**Architecture:**
```python
class AutoGPT:
    def run(self, goal):
        task_list = [goal]

        while task_list:
            # Get next task
            task = task_list.pop(0)

            # Execute
            result = self.execute_task(task)

            # Generate new tasks based on result
            new_tasks = self.create_tasks(result, goal)
            task_list.extend(new_tasks)

            # Reflect
            self.save_to_memory(task, result)
```

**Pros:**
- ✓ Fully autonomous
- ✓ Long-term memory (vector DB)
- ✓ Internet access, file system access

**Cons:**
- ✗ Unpredictable behavior
- ✗ Gets stuck in loops
- ✗ Very expensive (uncontrolled LLM calls)
- ✗ Needs human supervision

**Best for**: Open-ended exploration, research tasks

**3. LangGraph**

**What it is**: LangChain's graph-based agent framework

**Key idea**: Agent as state machine
```python
from langgraph.graph import StateGraph

# Define state
class AgentState(TypedDict):
    messages: List[Message]
    current_task: str
    completed: bool

# Define graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("planner", plan_task)
workflow.add_node("executor", execute_task)
workflow.add_node("reviewer", review_result)

# Add edges (control flow)
workflow.add_edge("planner", "executor")
workflow.add_conditional_edges(
    "reviewer",
    lambda state: "end" if state["completed"] else "planner"
)

# Compile
agent = workflow.compile()

# Run
result = agent.invoke({"messages": [user_message]})
```

**Pros:**
- ✓ Explicit control flow (easier to debug)
- ✓ Cyclical workflows supported
- ✓ Persistent state
- ✓ Human-in-the-loop built-in

**Cons:**
- ✗ More verbose than LangChain
- ✗ Steeper learning curve

**Best for**: Production agents, complex workflows

**4. CrewAI**

**What it is**: Multi-agent collaboration framework

**Architecture:**
```python
from crewai import Agent, Task, Crew

# Define agents with roles
researcher = Agent(
    role="Researcher",
    goal="Find accurate information",
    backstory="Expert researcher with 10 years experience",
    tools=[search_tool, scrape_tool]
)

writer = Agent(
    role="Writer",
    goal="Write engaging content",
    backstory="Professional content writer",
    tools=[write_tool]
)

# Define tasks
research_task = Task(
    description="Research topic X",
    agent=researcher
)

writing_task = Task(
    description="Write article based on research",
    agent=writer,
    context=[research_task]  # Depends on research
)

# Create crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process="sequential"
)

# Execute
result = crew.kickoff()
```

**Pros:**
- ✓ Built for multi-agent systems
- ✓ Role-based design is intuitive
- ✓ Task dependencies handled automatically

**Cons:**
- ✗ Newer, less mature
- ✗ Fewer integrations

**Best for**: Multi-agent applications

**5. Microsoft AutoGen**

**What it is**: Multi-agent conversation framework

```python
from autogen import AssistantAgent, UserProxyAgent

# Create agents
assistant = AssistantAgent(
    name="assistant",
    llm_config={"model": "gpt-4"}
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding"}
)

# Start conversation
user_proxy.initiate_chat(
    assistant,
    message="Plot a chart of nvidia stock price YTD"
)

# Assistant writes code
# User proxy executes code
# They iterate until task complete
```

**Pros:**
- ✓ Code execution built-in
- ✓ Conversation-oriented
- ✓ Human-in-the-loop options

**Cons:**
- ✗ Limited to conversational agents
- ✗ Steep learning curve

**Best for**: Code generation, data analysis agents

**6. Semantic Kernel (Microsoft)**

**What it is**: Enterprise-focused agent framework (.NET/Python)

```python
import semantic_kernel as sk

kernel = sk.Kernel()

# Add plugins (tools)
kernel.import_skill(WebSearchSkill(), "WebSearch")
kernel.import_skill(MathSkill(), "Math")

# Define semantic functions (prompts)
summarize = kernel.create_semantic_function(
    "Summarize the following text: {{$input}}",
    max_tokens=100
)

# Orchestrate
result = await kernel.run_async(
    summarize,
    input_str=long_text
)
```

**Pros:**
- ✓ Enterprise-ready
- ✓ .NET first-class support
- ✓ Built for production
- ✓ Strong typing

**Cons:**
- ✗ Smaller community
- ✗ More verbose than LangChain

**Best for**: Enterprise .NET applications

**7. Haystack**

**What it is**: Framework for search and question-answering

```python
from haystack import Pipeline
from haystack.nodes import BM25Retriever, FARMReader

# Build RAG pipeline
pipeline = Pipeline()
pipeline.add_node(
    component=BM25Retriever(document_store=doc_store),
    name="Retriever",
    inputs=["Query"]
)
pipeline.add_node(
    component=FARMReader(model_name="deepset/roberta-base-squad2"),
    name="Reader",
    inputs=["Retriever"]
)

# Run
result = pipeline.run(query="What is machine learning?")
```

**Pros:**
- ✓ Best for RAG and search
- ✓ Production-ready
- ✓ Flexible pipelines

**Cons:**
- ✗ Not general-purpose agent framework
- ✗ Focused on retrieval use cases

**Best for**: RAG, document QA systems

**8. Langfuse (Monitoring)**

**What it is**: Observability for LLM applications

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Trace agent execution
trace = langfuse.trace(name="agent_task")

# Log each step
trace.span(name="planning").end(output=plan)
trace.span(name="execution").end(output=result)
trace.span(name="review").end(output=review)

# View in dashboard: costs, latency, errors
```

**Pros:**
- ✓ Great for debugging agents
- ✓ Cost tracking
- ✓ User feedback collection

**Not a framework**, but essential for production agents.

**Comparison table:**

| Framework | Best for | Complexity | Maturity |
|-----------|----------|------------|----------|
| LangChain | Prototyping | Medium | High |
| LangGraph | Production | High | Medium |
| AutoGPT | Autonomous agents | Low | Medium |
| CrewAI | Multi-agent | Medium | Low |
| AutoGen | Code generation | Medium | Medium |
| Semantic Kernel | Enterprise .NET | Medium | Medium |
| Haystack | RAG/Search | Medium | High |

**When to use a framework:**

✓ **Prototyping**: Frameworks speed up development
✓ **Common patterns**: ReAct, RAG, tool use
✓ **Team alignment**: Shared patterns and practices

✗ **Simple use case**: Direct API calls may be simpler
✗ **Performance critical**: Frameworks add overhead
✗ **Custom requirements**: May fight framework abstractions

**Building without a framework:**

```python
# Simple custom agent
class SimpleAgent:
    def __init__(self, tools):
        self.tools = tools

    def run(self, task):
        # 1. Plan
        plan = llm(f"Create a plan for: {task}")

        # 2. Execute steps
        for step in parse_plan(plan):
            if is_tool_call(step):
                result = self.execute_tool(step)
            else:
                result = llm(step)

        # 3. Return result
        return result

# Pros: Full control, minimal overhead
# Cons: More code to maintain
```

**Production recommendations:**

**For startups:**
- Start: LangChain (fast prototyping)
- Scale: LangGraph (more control)

**For enterprises:**
- .NET shops: Semantic Kernel
- Python shops: LangGraph or custom

**For research:**
- AutoGPT, custom implementations

**For RAG:**
- Haystack or LlamaIndex

## Follow-up Questions
- When would you build a custom agent vs use a framework?
- How do frameworks handle agent failures?
- What's the performance overhead of using LangChain?
