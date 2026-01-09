# CNNs vs Transformers vs RNNs

**Category:** foundations
**Difficulty:** 3
**Tags:** architecture, comparison, fundamentals

## Question
Compare CNNs, RNNs, and Transformers. When would you use each?

## What to Cover
- **Set context by**: Framing these as three paradigms for sequence/spatial processing
- **Must mention**: Key properties of each (CNN: local/parallel, RNN: sequential/memory, Transformer: global/parallel), complexity tradeoffs, best use cases (images, language, time series)
- **Show depth by**: Discussing historical progression (RNN→Transformer for NLP), hybrid models (Conformer, ViT), and modern recommendations
- **Avoid**: Just describing each architecture without comparing them and explaining when to use which

## Answer
**Three paradigms for sequence/spatial processing:**

## **CNNs (Convolutional Neural Networks)**

**Architecture:**
```
Convolution: Local receptive field
  Each output depends on k neighbors (kernel size)

Stride: Downsampling
Pooling: Aggregate local regions

Hierarchical: Wider receptive fields in deeper layers
```

**Properties:**
- **Local**: Each neuron sees local patch
- **Translation invariant**: Same kernel everywhere
- **Parameter sharing**: Kernel weights reused
- **Parallel**: Can compute all positions at once

**Complexity:** O(n × k) where k = kernel size

**Best for:**
✓ Images (2D structure, local patterns)
✓ Audio (local spectral patterns)
✓ Short sequences with local dependencies

✗ Long-range dependencies (need many layers)
✗ Variable-length sequences

**Example:** ResNet for images, WaveNet for audio

## **RNNs (Recurrent Neural Networks)**

**Architecture:**
```
h_t = f(h_{t-1}, x_t)
y_t = g(h_t)

Sequential processing
Hidden state carries information
```

**Variants:**
- **Vanilla RNN**: Vanishing gradients
- **LSTM**: Gates for long-term memory
- **GRU**: Simpler than LSTM, similar performance

**Properties:**
- **Sequential**: Must process in order (t-1 before t)
- **Arbitrary length**: Natural for variable sequences
- **Memory**: Hidden state accumulates information

**Complexity:** O(n × d²) where d = hidden size

**Best for:**
✓ Sequential data (time series)
✓ Streaming (online processing)
✓ Small data (fewer parameters than Transformer)

✗ Long sequences (vanishing gradients)
✗ Parallelization (sequential bottleneck)
✗ Modern LLMs (transformers better)

**Example:** Speech recognition (older), language models (pre-transformer)

## **Transformers**

**Architecture:**
```
Attention: All-to-all connections
  Each token attends to all tokens

Self-attention: No recurrence, no convolution
Feed-forward: Per-position MLP
```

**Properties:**
- **Parallel**: Process all positions at once
- **Global**: Any position can affect any other (O(1) path)
- **Permutation invariant** (without position encoding)

**Complexity:** O(n² × d) attention, O(n × d²) FFN

**Best for:**
✓ Language modeling (current SOTA)
✓ Long-range dependencies
✓ Large datasets (scales well)
✓ Parallelizable training

✗ Very long sequences (O(n²) cost)
✗ Streaming (need full context for attention)

**Example:** GPT, BERT, T5, LLaMA

## **Comparison Table**

| Aspect | CNN | RNN | Transformer |
|--------|-----|-----|-------------|
| Receptive field | Local → grows | Global | Global |
| Parallelization | High | None | High |
| Memory | O(k) | O(1) | O(n) |
| Computation | O(n×k) | O(n) | O(n²) |
| Long-range | Weak | Medium | Strong |
| Inductive bias | Locality, translation | Sequential | None (needs pos enc) |
| Best for | Images | Streaming | Language |

## **Hybrid approaches:**

**Conformer (Speech):**
```
Convolution (local) + Attention (global)
Better than either alone for speech
```

**Vision Transformer (ViT):**
```
Patches as tokens (CNN-like preprocessing)
Then transformer
```

**Transformer with local attention:**
```
Attention window (like CNN receptive field)
Reduces O(n²) to O(n×k)
Example: Longformer
```

## **Historical progression:**

**Image:**
- CNNs dominant (2012-2020): ResNet, EfficientNet
- ViT emerging (2020+): Patch-based transformers
- Hybrid winning: ConvNext, Swin Transformer

**Sequence/Language:**
- RNNs (2010-2017): LSTM, GRU
- Transformers (2017-now): BERT, GPT, T5
- Transformers completely replaced RNNs for LLMs

**Speech:**
- RNNs + CNNs (2015-2019): DeepSpeech
- Transformers (2019-now): Whisper, Conformer

## **When to use what (2024):**

**Image classification:**
- CNN (efficient, proven): ResNet, EfficientNet
- Transformer (accuracy): ViT, Swin
- Hybrid (best): ConvNext

**Language modeling:**
- Transformer only: GPT, LLaMA
- RNN mostly dead for this

**Speech:**
- Transformer: Whisper
- Hybrid: Conformer

**Time series:**
- Depends: Transformers for long, CNNs for short
- RNNs still used in some domains

**Video:**
- Hybrid: 3D CNN + Transformer (VideoMAE, TimeSformer)

## Follow-up Questions
- Why did transformers replace RNNs for language?
- What inductive biases do CNNs have?
- Can transformers work without positional encodings?
