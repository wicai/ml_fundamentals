# GPT Architecture (Decoder-Only)

**Category:** modern_llm
**Difficulty:** 3
**Tags:** architecture, gpt, autoregressive

## Question
What defines a "decoder-only" architecture like GPT? How does it differ from BERT or T5?

## What to Cover
- **Set context by**: Framing the three main transformer variants (encoder-only, decoder-only, encoder-decoder)
- **Must mention**: Causal self-attention, autoregressive training objective, contrast with BERT (bidirectional) and T5 (encoder-decoder)
- **Show depth by**: Explaining why decoder-only won for LLMs (simplicity, scalability, in-context learning)
- **Avoid**: Only describing GPT without contrasting against BERT/T5 architecture differences

## Answer
**Decoder-Only (GPT, LLaMA, PaLM):**
```
Input → Token Embed + Pos Embed
     → Stack of Decoder Blocks (causal self-attention)
     → LM Head → Predict next token
```

**Key characteristics:**
1. **Causal self-attention**: Can only attend to past tokens
2. **Autoregressive**: Trained to predict next token P(x_t | x_<t)
3. **No encoder**: Single stack of transformer blocks
4. **Unified objective**: All training is next-token prediction

**vs BERT (Encoder-Only):**
- BERT: Bidirectional attention (see full context)
- BERT: Masked language modeling (predict random masked tokens)
- BERT: Good for classification, bad for generation

**vs T5 (Encoder-Decoder):**
- T5: Encoder (bidirectional) + Decoder (causal) + cross-attention
- T5: Trained on span corruption (predict masked spans)
- T5: Good for seq2seq tasks (translation, summarization)

**Why decoder-only won for LLMs?**

1. **Simplicity**: One architecture, one objective
2. **Scalability**: Easier to scale to 100B+ parameters
3. **In-context learning**: Autoregressive objective enables prompting
4. **Generality**: Can do classification (via generation), generation, reasoning

**Trade-off**: Less parameter-efficient for encoder tasks (like classification) vs BERT, but generality wins at scale.

## Follow-up Questions
- Can GPT-style models do bidirectional encoding?
- Why did encoder-decoder models (T5) not scale as well?
- How does prompting work with autoregressive models?
