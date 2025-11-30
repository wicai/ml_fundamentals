# ML Fundamentals Study System

Comprehensive study tool for ML/LLM interview preparation with spaced repetition.

## Overview

This system contains **~130 study items** organized for efficient learning:
- **80 concept Q&A**: Deep dives into ML/LLM fundamentals
- **30 quiz items**: Quick recall for SRS practice
- **12 deep explanations**: Practice articulating complex topics
- **5 derivations**: Mathematical fundamentals

## Content Coverage

### Transformers & Attention (~5 hours)
- Attention mechanisms, multi-head, self/cross-attention
- Positional encodings (RoPE, sinusoidal, learned)
- Layer normalization, residual connections
- Causal masking, transformer blocks
- Efficient attention (Flash, sparse, MQA/GQA)

### Modern LLM Architecture (~4 hours)
- GPT (decoder-only) architecture
- Tokenization (BPE, SentencePiece)
- KV caching, inference optimizations
- Quantization (4-bit, 8-bit)
- Model compression techniques

### Training LLMs (~3 hours)
- Pretraining objectives
- Mixed precision (FP16, BF16)
- Gradient accumulation, clipping
- AdamW optimizer, learning rate schedules
- Data filtering and quality

### RLHF & Alignment (~4 hours)
- Supervised fine-tuning (SFT)
- Reward model training
- PPO algorithm for RLHF
- DPO (Direct Preference Optimization)
- Constitutional AI, safety

### Distributed Training (~3 hours)
- Data parallelism (DDP, FSDP/ZeRO)
- Tensor parallelism, pipeline parallelism
- Activation checkpointing
- Hybrid parallelism strategies

### Foundations (~2 hours)
- Backpropagation, gradient descent variants
- Activation functions (GELU, SiLU)
- Regularization (dropout, weight decay)
- Initialization, batch/layer normalization
- Cross-entropy loss, softmax

### Evaluation & Scaling (~2 hours)
- Perplexity, benchmarks (MMLU, GSM8K)
- Few-shot learning, chain-of-thought
- Scaling laws (Chinchilla)
- Emergence, hallucination

### Advanced Topics (~5 hours)
- RAG (Retrieval-Augmented Generation)
- LoRA, PEFT, fine-tuning strategies
- Prompt engineering, function calling
- Multimodal models (CLIP, GPT-4V)
- Agents, model editing
- MoE (Mixture of Experts), Mamba/SSMs
- Serving infrastructure, API design

## Quick Start

### Run a Study Session

```bash
# Default: 10 items (mixed types)
python ml_fundamentals/study.py

# Specific number of items
python ml_fundamentals/study.py -n 20

# Focus on specific type
python ml_fundamentals/study.py -t concepts  # concepts, quiz, deep, or derive

# View statistics
python ml_fundamentals/study.py --stats
```

### Study Flow

The system uses **spaced repetition** (SRS) to optimize retention:

1. **Item selection**: Prioritizes items you haven't seen or got wrong
2. **Study**: Review question, think through answer, reveal solution
3. **Self-assessment**: Rate yourself 1-3:
   - `1`: No idea (review soon)
   - `2`: Partial understanding (review moderately soon)
   - `3`: Got it (review much later)
4. **Adaptive scheduling**: Items you know well appear less frequently

### Recommended Schedule

**Week 1-2: Foundations & Transformers**
- Study 10-15 items/day
- Focus on concepts (70%), quiz (30%)
- Start with fundamentals, then transformers

**Week 3-4: Modern LLMs & Training**
- Study 10-15 items/day
- Mix: concepts (60%), quiz (30%), deep (10%)
- Cover modern architectures, training techniques

**Week 5-6: RLHF, Distributed, Advanced**
- Study 10-15 items/day
- Mix: concepts (50%), quiz (30%), deep (15%), derive (5%)
- Cover alignment, serving, specialization

**Ongoing: Spaced repetition**
- Review 10-20 items/day
- System automatically prioritizes overdue items
- Focus on quiz items for fast recall

## File Structure

```
ml_fundamentals/
├── study.py              # Main study tool
├── .study_state.json     # Your progress (auto-generated)
├── README.md             # This file
├── concepts/             # 80 detailed Q&A
│   ├── c001_attention_mechanism.md
│   ├── c002_self_attention.md
│   └── ...
├── quiz/                 # 30 quick recall items
│   ├── q001_attention_formula.md
│   ├── q002_gpt_architecture_type.md
│   └── ...
├── deep/                 # 12 explanation practice
│   ├── d001_deep_explanation.md
│   └── ...
└── derive/              # 5 mathematical derivations
    ├── v001_derivation.md
    └── ...
```

## Item Format

Each study item is a markdown file:

```markdown
# Title

**Category:** transformers | training | rlhf | etc
**Difficulty:** 1-5
**Tags:** attention, architecture, optimization

## Question
Clear question that might appear in interview

## Answer
Detailed answer with:
- Key concepts
- Formulas (when relevant)
- Examples
- Common gotchas
- Practical implications

## Follow-up Questions
- Related questions interviewers might ask
```

## Study Tips

### For Concepts
1. Read question, think through answer before revealing
2. Note what you missed or didn't know
3. Pay attention to "gotchas" and "when to use"
4. Connect to other concepts you've learned

### For Quiz Items
1. Quick recall - aim for <30 seconds
2. Perfect for review sessions
3. Use these to maintain knowledge

### For Deep Explanations
1. Practice explaining out loud
2. Pretend you're teaching someone
3. Use whiteboard or paper to draw diagrams
4. Aim for 5-10 minute explanations

### For Derivations
1. Work through with pen and paper
2. Don't look at solution immediately
3. Understand each step, not just memorize
4. Connect math to intuition

## Customization

### Adjust Study Mix

Edit `study.py` to change default weights:

```python
weights = {
    'concepts': 0.7,  # 70% concepts
    'quiz': 0.2,      # 20% quiz
    'deep': 0.05,     # 5% deep
    'derive': 0.05    # 5% derive
}
```

### Add Your Own Items

Create new markdown files following the format above. The system will automatically include them.

### Reset Progress

```bash
rm ml_fundamentals/.study_state.json
```

## Interview Prep Strategy

### 1-2 Weeks Before Interview
- **Focus**: High-priority concepts, common questions
- **Routine**: 30-60 min/day on quiz + concepts
- **Goal**: Cover all 80 concepts at least once

### 3-7 Days Before Interview
- **Focus**: Deep explanations, derivations
- **Routine**: Practice explaining topics out loud
- **Goal**: Articulate 10-15 key topics fluently

### 1-2 Days Before Interview
- **Focus**: Quick review, quiz items only
- **Routine**: 30 min rapid-fire quiz review
- **Goal**: Refresh memory, build confidence

### Day of Interview
- **Focus**: Light review of 5-10 key concepts
- **Avoid**: Cramming new material
- **Goal**: Warm up, stay confident

## Progress Tracking

View your statistics:

```bash
python ml_fundamentals/study.py --stats
```

Shows:
- Total items studied
- Total sessions completed
- Recent session performance
- Items due for review

## Tips for Success

1. **Consistency > intensity**: 30 min daily beats 3 hours once a week
2. **Active recall**: Try to answer before revealing
3. **Spaced repetition**: Trust the system's scheduling
4. **Connect concepts**: Link new concepts to what you know
5. **Practice explaining**: Verbal explanation reveals gaps
6. **Review follow-ups**: Don't skip follow-up questions
7. **Take notes**: Write down what you struggle with

## Troubleshooting

**Items too easy/hard?**
- Focus on specific difficulty levels by editing code
- System learns your performance over time

**Want more items on topic X?**
- Add your own markdown files to appropriate directory
- System auto-detects new files

**Forgot everything?**
- Normal! Spaced repetition brings it back
- Items you struggle with appear more frequently

**Not enough time?**
- Do 5 items/day minimum
- Focus on quiz items for fast review
- Quality > quantity

## Advanced Features

The study tool tracks:
- Last seen date for each item
- Your performance history
- Optimal review intervals (SRS algorithm)
- Streak of correct answers

Items you consistently get right appear less often. Items you struggle with appear more frequently.

## Contributing

Feel free to:
- Add new study items
- Fix errors in existing content
- Share your study strategy
- Suggest improvements

## Good Luck!

Remember: The goal isn't to memorize everything, but to build **deep understanding** and **quick recall** of fundamentals. The best interview responses combine:
- Clear explanations (practice with deep items)
- Quick facts (practice with quiz items)
- Mathematical rigor (practice with derivations)
- Practical intuition (covered in concepts)

You've got this! 🚀
