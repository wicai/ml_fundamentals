# Agent State Management

**Category:** agents
**Difficulty:** 3
**Tags:** agents, state, architecture

## Question
How do agents manage state across multiple steps and conversations? What are the key design patterns?

## Answer
**State** = Information the agent needs to remember across steps/time.

**Types of state:**

**1. Conversation state**
```python
{
    "messages": [
        {"role": "user", "content": "Book a flight to NYC"},
        {"role": "assistant", "content": "What dates?"},
        {"role": "user", "content": "Next Monday"}
    ],
    "context_tokens": 250
}
```

**2. Task state**
```python
{
    "task_id": "123",
    "goal": "Book flight to NYC",
    "status": "in_progress",
    "steps_completed": 2,
    "steps_remaining": 3,
    "current_step": "comparing_prices",
    "data": {
        "destination": "NYC",
        "dates": "2024-02-05",
        "flight_options": [...]
    }
}
```

**3. User state (persistent)**
```python
{
    "user_id": "user_456",
    "preferences": {
        "airline": "Delta",
        "seat": "aisle",
        "budget": "under_500"
    },
    "history": {
        "past_bookings": [...],
        "frequent_destinations": ["NYC", "SF"]
    }
}
```

**State management patterns:**

**1. Stateless (each call independent)**
```python
def agent_call(user_message):
    # No state maintained between calls
    # Everything in the message
    response = llm(user_message)
    return response

# Simple but limited
# Can't remember conversation
```

**2. In-memory state**
```python
class StatefulAgent:
    def __init__(self):
        self.state = {
            "conversation": [],
            "current_task": None,
            "data": {}
        }

    def process(self, user_input):
        # Update state
        self.state["conversation"].append({
            "role": "user",
            "content": user_input
        })

        # Use state in processing
        response = self.generate_response(
            user_input,
            conversation_history=self.state["conversation"]
        )

        # Update state
        self.state["conversation"].append({
            "role": "assistant",
            "content": response
        })

        return response
```

**Pros**: Fast, simple
**Cons**: Lost when process restarts, can't scale across servers

**3. Persisted state (database)**
```python
class PersistentAgent:
    def __init__(self, session_id):
        self.session_id = session_id
        self.state = self.load_state()  # Load from DB

    def load_state(self):
        return db.get(f"session:{self.session_id}") or {}

    def save_state(self):
        db.set(f"session:{self.session_id}", self.state)

    def process(self, user_input):
        # Load state
        self.state = self.load_state()

        # Process
        response = self.generate_response(user_input, self.state)

        # Update and save
        self.update_state(response)
        self.save_state()

        return response
```

**Pros**: Survives restarts, works across servers
**Cons**: Slower (DB latency), more complex

**4. Event sourcing**
```python
class EventSourcedAgent:
    """Store all events, rebuild state by replaying them"""

    def __init__(self, session_id):
        self.session_id = session_id
        self.events = []

    def load_events(self):
        return db.get_all_events(self.session_id)

    def rebuild_state(self):
        """Replay all events to get current state"""
        state = {}
        for event in self.load_events():
            state = self.apply_event(state, event)
        return state

    def process(self, user_input):
        # Rebuild current state from events
        state = self.rebuild_state()

        # Generate response
        response = self.generate_response(user_input, state)

        # Save new event
        event = {
            "type": "message",
            "data": {"role": "user", "content": user_input},
            "timestamp": now()
        }
        db.append_event(self.session_id, event)

        return response
```

**Pros**: Full audit trail, can replay/debug
**Cons**: More complex, slower for long sessions

**5. Tiered state (hot + cold storage)**
```python
class TieredStateAgent:
    def __init__(self):
        # Hot: Recent messages (in memory)
        self.hot_state = {
            "recent_messages": []  # Last 10 messages
        }

        # Warm: Current session (Redis)
        self.warm_state_key = f"session:{session_id}"

        # Cold: Historical (PostgreSQL)
        self.cold_storage = database

    def get_context(self):
        # Combine tiered state
        context = {
            "recent": self.hot_state["recent_messages"],
            "session": redis.get(self.warm_state_key),
            "user_history": self.cold_storage.get_user_history()
        }
        return context
```

**State lifecycle:**

```python
class AgentStateMachine:
    """Agent as a state machine"""

    STATES = ["idle", "planning", "executing", "waiting_feedback", "done"]

    def __init__(self):
        self.current_state = "idle"
        self.data = {}

    def transition(self, event):
        # State transitions
        transitions = {
            ("idle", "new_task"): "planning",
            ("planning", "plan_ready"): "executing",
            ("executing", "step_complete"): "executing",
            ("executing", "needs_input"): "waiting_feedback",
            ("waiting_feedback", "input_received"): "executing",
            ("executing", "task_complete"): "done",
        }

        new_state = transitions.get((self.current_state, event))

        if new_state:
            self.current_state = new_state
        else:
            raise ValueError(f"Invalid transition: {self.current_state} + {event}")

    def run(self):
        while self.current_state != "done":
            if self.current_state == "planning":
                self.plan()
                self.transition("plan_ready")

            elif self.current_state == "executing":
                result = self.execute_step()

                if result.complete:
                    self.transition("task_complete")
                elif result.needs_input:
                    self.transition("needs_input")
                else:
                    self.transition("step_complete")

            elif self.current_state == "waiting_feedback":
                feedback = self.wait_for_user()
                self.transition("input_received")
```

**State size management:**

```python
class BoundedState:
    def __init__(self, max_tokens=8000):
        self.messages = []
        self.max_tokens = max_tokens

    def add_message(self, msg):
        self.messages.append(msg)

        # Prune if too large
        while self.count_tokens() > self.max_tokens:
            # Strategy 1: Drop oldest
            self.messages.pop(0)

            # Strategy 2: Summarize old messages
            if len(self.messages) > 20:
                old_batch = self.messages[:10]
                summary = llm.summarize(old_batch)
                self.messages = [summary] + self.messages[10:]

    def get_context(self):
        return self.messages
```

**Distributed state (multi-server):**

```python
# Use Redis for shared state across servers
import redis

class DistributedAgent:
    def __init__(self, session_id):
        self.redis = redis.Redis()
        self.session_id = session_id

    def get_state(self):
        state_json = self.redis.get(f"agent_state:{self.session_id}")
        return json.loads(state_json) if state_json else {}

    def update_state(self, updates):
        # Atomic update with lock
        with self.redis.lock(f"lock:{self.session_id}"):
            state = self.get_state()
            state.update(updates)
            self.redis.set(
                f"agent_state:{self.session_id}",
                json.dumps(state),
                ex=3600  # Expire after 1 hour
            )
```

**State versioning:**

```python
class VersionedState:
    """Track state versions for rollback"""

    def __init__(self):
        self.versions = []
        self.current_version = 0

    def checkpoint(self, state):
        """Save a version"""
        self.versions.append(copy.deepcopy(state))
        self.current_version = len(self.versions) - 1

    def rollback(self, steps=1):
        """Revert to previous version"""
        self.current_version = max(0, self.current_version - steps)
        return self.versions[self.current_version]

    def get_state(self):
        return self.versions[self.current_version]
```

**Production patterns:**

**OpenAI Assistants:**
```python
# Threads = persistent state
thread = client.beta.threads.create()

# Add messages to thread
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Help me book a flight"
)

# State persists across API calls
# No need to send full history each time
```

**LangChain memory:**
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()

# Automatically manages conversation state
agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory  # Handles state for you
)
```

**State anti-patterns:**

**❌ Global state:**
```python
# Bad: Shared across all users!
CURRENT_TASK = None

def process(user_input):
    global CURRENT_TASK
    # Race conditions, mixed state
```

**❌ Unvalidated state:**
```python
# Bad: Trusting state blindly
state = load_state()
amount = state["amount"]  # Could be None, invalid, etc.
charge_user(amount)  # Crash or wrong charge!
```

**✅ Validated, scoped state:**
```python
class ValidatedState:
    def __init__(self, session_id):
        self.session_id = session_id  # Scoped to session

    def get_amount(self):
        amount = self.state.get("amount")

        # Validate
        if amount is None or amount <= 0:
            raise ValueError("Invalid amount in state")

        return amount
```

**Key decisions:**

| Factor | In-memory | Database | Event sourcing |
|--------|-----------|----------|----------------|
| Speed | Fast | Medium | Slow |
| Persistence | No | Yes | Yes |
| Scalability | Poor | Good | Good |
| Debuggability | Poor | Medium | Excellent |
| Complexity | Low | Medium | High |

## Follow-up Questions
- How do you migrate agent state when updating the schema?
- What's the tradeoff between stateless and stateful agents?
- How do you handle state conflicts in multi-user agents?
