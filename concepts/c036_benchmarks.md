# LLM Evaluation Benchmarks

**Category:** evaluation
**Difficulty:** 2
**Tags:** evaluation, benchmarks, metrics

## Question
What are the main benchmarks for evaluating LLMs and what do they measure?

## What to Cover
- **Set context by**: Categorizing benchmarks by type (knowledge, reasoning, coding, safety, aggregate)
- **Must mention**: Key benchmarks (MMLU, GSM8K, HumanEval, TruthfulQA), what each measures, approximate SOTA numbers
- **Show depth by**: Discussing problems with benchmarks (contamination, saturation, Goodhart's law) and the trend toward human evaluation
- **Avoid**: Just listing benchmarks without explaining what they measure or their limitations

## Answer
**Categories of benchmarks:**

**1. General Knowledge & Reasoning**

**MMLU** (Massive Multitask Language Understanding)
- 57 subjects (math, history, law, etc.)
- Multiple choice
- Measures: Breadth of knowledge
- GPT-4: ~86%, GPT-3.5: ~70%

**HellaSwag / PIQA**
- Common sense reasoning
- Sentence completion
- Measures: Physical/social reasoning

**2. Math & Coding**

**GSM8K** (Grade School Math)
- 8K math word problems
- Measures: Arithmetic reasoning
- GPT-4: ~92%, GPT-3.5: ~57%

**MATH**
- Competition-level math
- Much harder than GSM8K
- GPT-4: ~52%

**HumanEval**
- Code generation from docstrings
- Measures: Coding ability
- GPT-4: ~67%, GPT-3.5: ~48%

**3. Reading Comprehension**

**SQuAD** / **NaturalQuestions**
- Answer questions about passages
- Measures: Reading comprehension

**4. Alignment & Safety**

**TruthfulQA**
- Questions designed to elicit false answers
- Measures: Truthfulness vs memorized misinformation
- GPT-4: ~60%

**BBQ** (Bias Benchmark)
- Measures social biases

**5. Aggregate Benchmarks**

**HELM** (Holistic Evaluation of Language Models)
- 42 scenarios, 7 metrics
- Comprehensive but complex

**Lmsys Chatbot Arena**
- Human preference (Elo ratings)
- Measures: Real-world usefulness

**Problems with benchmarks:**

1. **Contamination**: Test sets leaked into training data
2. **Saturation**: GPT-4 near ceiling on many benchmarks
3. **Goodhart's law**: Optimizing benchmarks ≠ useful models
4. **Multiple choice**: Can game with probability tricks
5. **Not representative**: Missing creative tasks, long-form generation

**Modern trend**: Moving toward human evaluation (e.g., Chatbot Arena) and task-specific evals.

## Follow-up Questions
- What is benchmark contamination?
- Why is MMLU considered important?
- How do you design a good benchmark?
