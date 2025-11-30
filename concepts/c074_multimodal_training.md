# Multimodal Training & Architecture

**Category:** modern_llm
**Difficulty:** 4
**Tags:** multimodal, training, architecture

## Question
How are multimodal models (text, image, audio) trained and architected?

## Answer
**Multimodal model**: Process and generate multiple modalities (text, image, audio, video).

**Main paradigms:**

**1. Dual-encoder (Contrastive - CLIP, ALIGN)**

**Architecture:**
```
Image → Image Encoder → image_embedding
Text → Text Encoder → text_embedding

Shared embedding space (512-1024 dim)
```

**Training:**
```
Data: (image, caption) pairs

Loss: Contrastive
  Maximize: sim(image_i, text_i)
  Minimize: sim(image_i, text_j) for j ≠ i

Batch of N pairs → N² similarities
InfoNCE loss
```

**Example (CLIP):**
```
400M image-text pairs from internet
Image encoder: ViT-L/14 or ResNet
Text encoder: Transformer (63M-123M params)

Zero-shot classification, retrieval, image generation
```

**2. Encoder-decoder (Generative - GPT-4V, LLaVA, Flamingo)**

**Architecture:**
```
Visual encoder → Visual tokens
Text tokens
LLM processes [visual tokens] + [text tokens] → text output
```

**Training stages:**

**Stage 1: Modality alignment**
```
Freeze: Vision encoder, LLM
Train: Projection layer (visual → text space)

Data: Image-caption pairs (millions)
Objective: Predict caption given image

Quick (hours), aligns modalities
```

**Stage 2: Instruction tuning**
```
Freeze: Vision encoder
Train: LLM (full or LoRA)

Data: VQA, detailed descriptions, instructions (100K-1M)
Objective: Follow instructions about images

Longer (days), teaches task following
```

**Example (LLaVA architecture):**
```
CLIP ViT-L/14 (frozen) → 256 visual tokens
  ↓
Linear projection (trainable) → 4096-d embeddings
  ↓
Llama-2 7B (LoRA fine-tuned) → text generation
```

**3. Unified (Any-to-any - Gemini, CoDi)**

**Goal**: One model handles all modality combinations.

**Architecture:**
```
Modality encoders:
  - Image → ViT
  - Audio → AST (Audio Spectrogram Transformer)
  - Video → VideoMAE

Shared latent space

Modality decoders:
  - Text → LLM
  - Image → Diffusion model
  - Audio → Vocoder

Can go: image → text, text → image, audio → text, etc.
```

**Data challenges:**

**1. Paired data:**
```
Image-text: Abundant (billions from web)
Image-audio: Rare
Video-text-audio: Very rare

Solution: Weak supervision, automatic captioning
```

**2. Alignment:**
```
Image captioned as "A cat"
But image has complex scene

Noisy correspondence
Solution: Filter, rerank, use better captions
```

**3. Balance:**
```
Text data: Trillions of tokens
Image-text: Billions of pairs
Video-text: Millions of pairs

How to mix? Active research
```

**Training objectives:**

**Contrastive:**
```
Align modalities in shared space
Good for retrieval, not generation
```

**Autoregressive:**
```
Predict next token (text) or next patch (image)
Unified objective across modalities
```

**Masked:**
```
Mask parts of input, predict masked parts
MAE (images), BERT (text), VideoMAE (video)
```

**Diffusion:**
```
For image/audio generation
Denoise noisy inputs
```

**Modality-specific challenges:**

**Images:**
```
High-dimensional (224×224×3 = 150K pixels)
→ Compress to 256-576 tokens via ViT

Patch size: Trade-off between detail and efficiency
```

**Audio:**
```
Temporal (1 second = 16K samples at 16kHz)
→ Spectrogram + transformer
→ 100-200 tokens per second

Speech: Whisper, wav2vec2
Music: Jukebox, MusicLM
```

**Video:**
```
Extremely high-dimensional (30 FPS × resolution)
→ Sample sparse frames
→ Hierarchical processing

Video ViT: Process 8-16 frames
```

**Benchmarks:**

- **VQA**: Visual question answering
- **COCO Captions**: Image captioning
- **MMLU-Vision**: Visual reasoning
- **AudioCaps**: Audio captioning

**State-of-the-art (2024):**

**GPT-4V (OpenAI):**
- Text + image input → text output
- Best quality, closed-source

**Gemini (Google):**
- Natively multimodal (trained together)
- Text, image, audio, video

**LLaVA (open-source):**
- Competitive with GPT-4V on some tasks
- 7B-34B parameters

**CogVLM (open-source):**
- Visual expert attention layers
- Strong visual grounding

**Modern trends:**

- End-to-end training (not just alignment)
- Higher resolution images (1024×1024+)
- Video understanding
- Any-to-any modality conversion

## Follow-up Questions
- How do you align vision and language embeddings?
- What's the difference between CLIP and LLaVA?
- How much paired data is needed for multimodal training?
