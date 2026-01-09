# LLM Interpretability & Mechanistic Interpretability

**Category:** evaluation
**Difficulty:** 4
**Tags:** interpretability, analysis, safety

## Question
How do researchers understand what LLMs are doing internally? What is mechanistic interpretability?

## What to Cover
- **Set context by**: Explaining why interpretability matters (debugging, alignment, trust)
- **Must mention**: Main approaches (attention visualization, activation patching, probing, mechanistic interp), key findings (induction heads, superposition)
- **Show depth by**: Mentioning specific techniques (sparse autoencoders, activation steering) and limitations (scalability, polysemantic neurons)
- **Avoid**: Only describing attention visualization without covering mechanistic interpretability (the deeper approach)

## Answer
**Goal**: Understand how LLMs work internally, not just inputs/outputs.

**Why it matters:**
- Debug failures
- Improve alignment
- Detect deception
- Build trust

**Approaches:**

**1. Attention visualization:**
```
Which tokens does the model attend to?

Visualization:
  Query: "The cat"
  Attends to: "The" (0.8), "cat" (0.2)

Shows: Syntactic relationships, coreference
Limitation: Doesn't explain what's computed
```

**2. Activation patching / Causal tracing:**
```
Change activation at layer L, position i
Measure effect on output

Example:
  "The Eiffel Tower is in [MASK]"
  Which layers encode "Eiffel Tower → Paris"?

Ablate different layers → find critical ones
```

**3. Logit lens / Tuned lens:**
```
Decode hidden states at each layer:
  h_layer_10 → vocab distribution

Shows: What model "thinks" at intermediate layers

Finding: Models decide answer early, refine later
```

**4. Probing classifiers:**
```
Train small classifier on hidden states to predict property

h → Classifier → "Is this token a noun?"

If high accuracy: Information is linearly accessible
Doesn't mean model uses it that way
```

**5. Mechanistic interpretability (detailed circuit analysis):**

**Goal**: Reverse-engineer exact algorithms

**Example - Indirect Object Identification:**
```
"When Mary and John went to the store, John gave a drink to"
Model predicts: "Mary"

Circuit:
  1. Attention head finds subject-verb agreement
  2. Another head does coreference resolution
  3. MLP composes them
  4. Output head selects "Mary"

Fully characterized algorithm
```

**Key techniques:**

**Activation steering:**
```
Find direction in activation space for concept
  e.g., "honesty" direction

Add this direction → model more honest
Subtract → model less honest

Shows: Concepts are represented as directions
```

**Sparse autoencoders (SAEs):**
```
Problem: Superposition - neurons represent multiple concepts

SAE: Find sparse, interpretable features
  h = SAE(h)  # Sparse code

Each SAE neuron = one interpretable concept
  Neuron 472: Activates on "Golden Gate Bridge"
```

**Circuit discovery:**
```
1. Identify task (e.g., greater-than comparison)
2. Ablate components (heads, MLPs, layers)
3. Find minimal circuit that solves task
4. Understand algorithm

Example: Modular arithmetic circuit in transformers
```

**Findings:**

**Induction heads (GPT-2, others):**
```
Pattern: A B ... A [?]
Predicts: B

Two-head circuit:
  Head 1: Finds previous "A"
  Head 2: Copies next token ("B")

Fundamental for in-context learning
```

**Factual recall:**
```
"The Eiffel Tower is in" → "Paris"

Knowledge stored in MLP layers (mid-network)
Retrieved via pattern matching in attention
```

**Superposition:**
```
d_model = 512, but represents 10,000+ features

How? Features as directions, not neurons
  Allows more features than dimensions

Challenge for interpretability
```

**Tools & frameworks:**

- **TransformerLens**: Library for mech interp (easy activation access)
- **Neuron2Graph**: Visualize circuits
- **CircuitsVis**: Interactive attention patterns
- **Anthropic's Influence Functions**: Which training examples affect outputs

**Limitations:**

1. **Scalability**: Hard to analyze 175B param models
2. **Superposition**: Features entangled
3. **Polysemantic neurons**: One neuron = multiple concepts
4. **Emergence**: Circuits interact in complex ways

**Open problems:**

- Understanding chain-of-thought reasoning circuits
- How does in-context learning work mechanistically?
- Can we predict failures from internals?

## Follow-up Questions
- What are induction heads and why do they matter?
- How does mechanistic interpretability differ from saliency maps?
- Can you modify model behavior by editing activations?
