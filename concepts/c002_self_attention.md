# Self-Attention vs Cross-Attention

**Category:** transformers
**Difficulty:** 3
**Tags:** attention, architecture

## Question
What's the difference between self-attention and cross-attention? When is each used?

## What to Cover
- **Set context by**: Explaining that attention can operate within or across sequences
- **Must mention**: Q, K, V source differences, concrete use cases (encoder self-attention, decoder self-attention, encoder-decoder cross-attention)
- **Show depth by**: Explaining why decoder-only models (GPT) don't need cross-attention
- **Avoid**: Giving abstract definitions without concrete architectural examples

## Answer
**Self-Attention:**
- Q, K, V all come from the same sequence
- Each position attends to all positions in the same sequence
- Used in: Transformer encoder/decoder blocks

**Cross-Attention:**
- Q comes from one sequence, K and V from another
- Used to combine information from two different sources
- Used in: Encoder-decoder models (decoder attends to encoder outputs)

**Example in Machine Translation:**
- **Encoder self-attention**: French sentence attends to itself
- **Decoder self-attention**: English sentence attends to itself (causal mask)
- **Cross-attention**: English decoder attends to French encoder outputs

**Modern LLMs (GPT):**
- Only use self-attention (decoder-only architecture)
- No cross-attention needed for autoregressive language modeling

## Follow-up Questions
- Why do decoder-only models not need cross-attention?
- What is causal/masked self-attention?
