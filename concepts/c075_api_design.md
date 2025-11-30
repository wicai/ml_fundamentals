# LLM API Design & Interface Patterns

**Category:** modern_llm
**Difficulty:** 2
**Tags:** api, deployment, interface

## Question
What are the key considerations and patterns for designing LLM APIs?

## Answer
**Goal**: Provide clean, efficient, safe interface to LLM.

**Core API patterns:**

**1. Completion API (basic)**
```
POST /v1/completions
{
  "prompt": "The capital of France is",
  "max_tokens": 50,
  "temperature": 0.7
}

Response:
{
  "text": "Paris, known for the Eiffel Tower...",
  "finish_reason": "length",
  "tokens_used": 50
}

Simple, stateless, one-shot
```

**2. Chat API (conversational)**
```
POST /v1/chat/completions
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is photosynthesis?"},
    {"role": "assistant", "content": "Photosynthesis is..."},
    {"role": "user", "content": "Can you explain more?"}
  ],
  "model": "gpt-4",
  "temperature": 0.7
}

Handles conversation context
Standard: OpenAI format
```

**3. Streaming API**
```
POST /v1/chat/completions
{
  "messages": [...],
  "stream": true
}

Response: Server-Sent Events (SSE)
data: {"choices": [{"delta": {"content": "The"}}]}
data: {"choices": [{"delta": {"content": " answer"}}]}
data: {"choices": [{"delta": {"content": " is"}}]}
...

Real-time token-by-token output
Better UX (see progress)
```

**Key parameters:**

**Model selection:**
```
"model": "gpt-4" / "gpt-3.5-turbo" / "claude-3-opus"

Version pinning:
  "gpt-4-0613" (specific checkpoint)
  vs "gpt-4" (latest version, changes over time)
```

**Generation control:**
```
"temperature": 0.0-2.0
  Lower = more deterministic

"top_p": 0.0-1.0 (nucleus sampling)
  Alternative to temperature

"max_tokens": int
  Limit output length

"stop": ["\n", "END"]
  Stop sequences

"frequency_penalty": -2.0 to 2.0
  Penalize repeated tokens

"presence_penalty": -2.0 to 2.0
  Penalize any repetition
```

**Advanced features:**

**1. Function calling:**
```
{
  "messages": [...],
  "functions": [
    {
      "name": "get_weather",
      "description": "Get weather for location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {"type": "string"}
        }
      }
    }
  ]
}

Model can return function call instead of text
```

**2. Logprobs:**
```
{
  "logprobs": true,
  "top_logprobs": 5
}

Returns:
  Token probabilities
  Alternative tokens considered
  Useful for confidence estimation
```

**3. Seed (deterministic):**
```
{
  "seed": 12345
}

Same seed + prompt → same output
Helpful for debugging, testing
```

**4. System fingerprint:**
```
Response includes:
  "system_fingerprint": "fp_abc123"

Identifies model version
Detect if model changed
```

**Rate limiting & quotas:**

**Rate limits:**
```
Requests per minute (RPM): 3,500
Tokens per minute (TPM): 90,000

Response headers:
  X-RateLimit-Limit-Requests: 3500
  X-RateLimit-Remaining-Requests: 3499
  X-RateLimit-Reset-Requests: 2024-01-01T00:00:00Z

Status 429: Too many requests
```

**Quota management:**
```
Organizational quotas
Per-user quotas
Budget limits (e.g., $100/month)
```

**Safety & moderation:**

**Input filtering:**
```
POST /v1/moderations
{
  "input": "User prompt here"
}

Returns:
{
  "flagged": true,
  "categories": {"hate": true, "violence": false, ...}
}

Block request if flagged
```

**Output filtering:**
```
After generation, moderate output
Regenerate if harmful
```

**Error handling:**

**Common errors:**
```
400 Bad Request: Invalid parameters
401 Unauthorized: Bad API key
429 Too Many Requests: Rate limit
500 Internal Server Error: Model failure
503 Service Unavailable: Overload

Include helpful error messages:
{
  "error": {
    "message": "Invalid temperature: must be 0-2",
    "type": "invalid_request_error",
    "param": "temperature",
    "code": "invalid_value"
  }
}
```

**Retries:**
```
Exponential backoff for 429, 500, 503
Max retries: 3-5
Timeout: 60s for long requests
```

**Pricing & billing:**

**Token-based:**
```
Input tokens: $0.03 / 1K
Output tokens: $0.06 / 1K (often 2× input)

Calculate:
  Input: 100 tokens
  Output: 200 tokens
  Cost: 100*0.03/1000 + 200*0.06/1000 = $0.015
```

**Caching (some providers):**
```
Repeated prompt prefixes cached → cheaper
"System prompt" cached automatically
```

**Best practices:**

1. **Validate inputs**: Prevent injection, check limits
2. **Stream for UX**: Better user experience
3. **Log prompts**: Debug, quality assurance
4. **Monitor costs**: Track token usage
5. **Handle errors**: Graceful fallbacks
6. **Set timeouts**: Don't wait forever
7. **Use function calling**: Structured outputs
8. **Version pin**: Avoid surprise changes

**OpenAI-compatible:**
- Standard format
- Many providers compatible (Anthropic, Anyscale, Together, vLLM)
- Easy to switch

## Follow-up Questions
- What's the difference between temperature and top_p?
- How do you handle rate limits in production?
- What's the advantage of streaming responses?
