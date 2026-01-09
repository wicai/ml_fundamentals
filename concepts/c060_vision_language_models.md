# Vision-Language Models

**Category:** modern_llm
**Difficulty:** 4
**Tags:** multimodal, vision, architecture

## Question
How do vision-language models like GPT-4V and CLIP work?

## What to Cover
- **Set context by**: Distinguishing the two paradigms (contrastive like CLIP, generative like GPT-4V)
- **Must mention**: CLIP architecture and contrastive training, generative VLM architecture (vision encoder + projector + LLM), training stages (alignment pretraining, instruction tuning)
- **Show depth by**: Discussing challenges (high-res images, hallucination) and mentioning specific models (LLaVA, CLIP)
- **Avoid**: Only describing one paradigm without comparing contrastive vs generative approaches

## Answer
**Vision-Language Models**: Process both images and text.

**Two main paradigms:**

**1. Contrastive (CLIP, ALIGN)**

**Training:**
```
Data: Image-text pairs from internet (billions)

Objective: Match images with correct captions

For each (image, text) pair:
  image_embedding = Vision_Encoder(image)
  text_embedding = Text_Encoder(text)

  Maximize: cosine_similarity(image_emb, text_emb)
  Minimize: cosine_similarity(image_emb, wrong_text_emb)

Batch of N pairs → N² similarity matrix
Positive pairs on diagonal, negatives off-diagonal
```

**Architecture (CLIP):**
```
Vision encoder: ViT (Vision Transformer) or ResNet
Text encoder: Transformer (BERT-like)

Both project to shared embedding space (512-1024 dim)
```

**Use cases:**
- Zero-shot image classification
- Image-text retrieval
- Image generation (as in Stable Diffusion, DALL-E)

**2. Generative (GPT-4V, LLaVA, Flamingo)**

**Architecture:**
```
1. Vision encoder: ViT or CLIP visual encoder
   Image → sequence of visual tokens/embeddings

2. Projector/Adapter:
   Map visual embeddings to LLM's embedding space

3. Language model:
   Process [visual tokens] + [text tokens]
   Generate text output

Example:
  Input: <image> + "What's in this image?"
  Output: "A cat sitting on a couch"
```

**Training strategies:**

**Stage 1: Alignment pretraining**
```
Freeze LLM, freeze vision encoder
Train only the projector

Data: Image-caption pairs
Objective: Next-token prediction on captions
```

**Stage 2: Instruction tuning**
```
Unfreeze LLM (or use LoRA)
Train on instruction-following data

Data: Visual question answering, detailed descriptions
Objective: Follow instructions about images
```

**Examples:**

**CLIP (OpenAI):**
- Vision: ViT-L/14
- Text: Transformer 12 layers
- Training: 400M image-text pairs
- Zero-shot classification competitive with supervised

**LLaVA (open-source GPT-4V alternative):**
```
Vision: CLIP ViT-L/14
Projector: MLP
LLM: Llama-2 7B/13B

Training:
  1. Pretrain projector on image captions (CC3M)
  2. Fine-tune on 158K instruction data (GPT-4 generated)
```

**GPT-4V (rumored architecture):**
- Similar approach but larger scale
- Vision encoder integrated into main model
- Trained on vast multimodal internet data

**Challenges:**

1. **High-res images**: ViT on 224×224 patches loses details
   - Solution: Patch-level processing, higher resolution encoding

2. **Alignment**: Visual and text spaces very different
   - Solution: Large-scale contrastive pretraining

3. **Hallucination**: Model describes things not in image
   - Solution: Grounding, retrieval-augmented generation

4. **Compute**: Images → many tokens (256-1024 visual tokens)
   - Solution: Compression, efficient attention

**Modern trends:**

- **Higher resolution**: 1024×1024 or adaptive
- **Video understanding**: Extend to temporal dimension
- **Any-to-any**: Unified model for vision, language, audio
- **Interleaved data**: Image-text-image-text sequences

## Follow-up Questions
- How does CLIP enable zero-shot classification?
- What's the difference between contrastive and generative VLMs?
- How do you evaluate vision-language models?
