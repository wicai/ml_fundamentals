# Mixture of Experts (MoE)

**Category:** modern_llm
**Difficulty:** 4
**Tags:** architecture, efficiency, scaling

## Question
What is Mixture of Experts and what are its advantages for LLMs?

## What to Cover
- **Set context by**: Explaining MoE as sparse computation (not all parameters used per token)
- **Must mention**: The routing mechanism (top-k experts), efficiency benefit (more params with less compute), examples (Mixtral, Switch Transformer, GLaM)
- **Show depth by**: Discussing challenges (load balancing, training instability, communication overhead) and their solutions
- **Avoid**: Only describing the architecture without explaining the efficiency benefits or challenges

## Answer
**MoE**: Replace dense feed-forward layers with multiple expert networks, route each token to subset of experts.

**Standard transformer block:**
```
FFN(x) = GELU(xW_1)W_2
All tokens go through same weights
```

**MoE transformer block:**
```
Router: scores = Softmax(xW_router)  # [num_experts]
Select top-k experts (k=1 or 2)

Output = Σ score_i * Expert_i(x)  # For selected experts

where each Expert_i is a separate FFN
```

**Example (8 experts, top-2 routing):**
```
Token "Paris": Routes to Expert 3 (0.7) and Expert 5 (0.3)
Token "eats": Routes to Expert 1 (0.6) and Expert 7 (0.4)

Output = 0.7*Expert3(x) + 0.3*Expert5(x)
```

**Key advantage: Sparse computation**
```
Dense model (7B params): All 7B params used per token
MoE model (56B params, 8 experts, top-2):
  Only 2/8 experts active → 14B params used per token

Total params: 56B (8× dense)
Compute per token: 14B (2× dense)

Result: 4× more parameters for 2× compute!
```

**Benefits:**

1. **Efficiency**: More parameters without proportional compute increase
2. **Specialization**: Experts learn different patterns
   - Expert 1: Code syntax
   - Expert 2: Math reasoning
   - Expert 3: Creative writing
3. **Scaling**: Can scale to trillions of parameters

**Challenges:**

1. **Load balancing**: Some experts overused, others underused
   - Solution: Auxiliary loss to encourage balanced routing

2. **Training instability**: Routing can collapse (all tokens → one expert)
   - Solution: Noise in routing, load balancing loss

3. **Communication overhead**: Experts split across GPUs
   - Solution: Expert parallelism (each GPU owns subset of experts)

4. **Fine-tuning**: Harder to fine-tune than dense models

**Examples:**

**Switch Transformer (Google, 2021)**
- 1.6T parameters, top-1 routing
- Better than dense models with same compute

**GLaM (Google, 2022)**
- 1.2T parameters, 64 experts, top-2 routing
- Matches GPT-3 with 1/3 compute

**Mixtral 8x7B (Mistral, 2023)**
- 47B total params, 8 experts, top-2 routing
- 13B active params per token
- Outperforms Llama 2 70B on many tasks

**GPT-4 (rumored)**
- Likely uses MoE (unconfirmed)

**Routing strategies:**

- **Learned router**: Trainable (standard)
- **Hash-based**: Deterministic, no load balancing issues
- **Expert choice**: Experts select tokens (instead of tokens selecting experts)

## Follow-up Questions
- How does expert parallelism work?
- What is the load balancing problem in MoE?
- Why doesn't MoE help as much for small models?
