# VideoLLaMA-Mini: A From-Scratch Implementation for Video Understanding

## Overview

This repository presents VideoLLaMA-Mini, a lightweight, from-scratch implementation of a video-to-text model inspired by modern Vision-Language Models (VLMs) such as VideoLLaMA, BLIP-2, and ViT-based architectures.

The primary goal of this project is not to achieve state-of-the-art performance, but to deeply understand and implement the core architectural modules involved in video understanding using transformers. The project is intentionally designed to be:

- Minimal
- CPU-friendly
- Fully readable and modifiable
- Educational rather than production-oriented

This implementation demonstrates how video frames, visual embeddings, cross-modal attention, and autoregressive text generation interact inside a modern vision-language pipeline.

## Project Objectives

The key objective of this project is:

**To understand the end-to-end implementation of different modules involved in video understanding using Vision Transformers and cross-modal attention mechanisms.**

Specifically, this project focuses on:

- Video frame sampling and preprocessing
- Spatial feature extraction from frames
- Temporal modeling of video content
- Cross-attention between visual and textual modalities
- Autoregressive text generation using causal masking
- Understanding how Vision Transformers and language models interact in VLMs

This repository serves as a **learning scaffold** for researchers and engineers who want to go beyond theory and gain hands-on experience with video-language models.

## Architecture Overview

The model follows a simplified **Video-LLaMA–style pipeline**, consisting of the following components:

```
Video Frames
   ↓
CNN-based Spatial Encoder
   ↓
Temporal Transformer Encoder
   ↓
Visual Tokens
   ↓
Q-Former (Cross-Attention)
   ↓
Text Tokens (with Positional Encoding)
   ↓
Causal Transformer Decoder
   ↓
Autoregressive Caption Generation
```

## Key Components
### 1. Dataset Handling (VideoCaptionDataset)

- Reads video–caption pairs from a text file
- Supports multi-line captions
- Uniformly samples a fixed number of frames from each video
- Normalizes and formats frames as tensors
- Tokenizes captions using GPT-2 tokenizer

**Design choice:**
Frame sampling is intentionally simple to highlight architectural understanding rather than dataset engineering.

### 2. Video Encoder (VideoEncoderMini)

This module converts raw video frames into a sequence of visual embeddings:
- CNN backbone extracts spatial features per frame
- Adaptive average pooling produces compact frame-level embeddings
- Transformer encoder models temporal dependencies across frames

This mirrors the idea of ViT-style temporal modeling, but with a CNN front-end for simplicity.

### 3. Positional Encoding

A classic sinusoidal positional encoding is used for text tokens, enabling the transformer to model word order without recurrence.

### 4. Q-Former (Cross-Attention)

The Q-Former block enables cross-modal interaction:
- Text tokens act as queries
- Visual tokens act as keys and values
- Multi-head attention fuses visual context into textual representations

This component is inspired by BLIP-2 and VideoLLaMA, where a lightweight module bridges vision and language.

### 5. Transformer Decoder with Causal Masking

- Uses stacked TransformerEncoderLayers with causal masking
- Enforces autoregressive generation
- Predicts the next token conditioned on:
  - Previously generated tokens
  - Video context

This allows the model to generate captions one token at a time, similar to GPT-style decoding.

### 6. Text Generation

- Starts generation from a neutral BOS token
- Uses top-k sampling and temperature scaling
- Stops automatically at the EOS token

This avoids trivial copying of ground-truth captions and ensures true autoregressive behavior.

## Training Strategy

- Teacher forcing with shifted input tokens
- Cross-entropy loss ignoring padding tokens
- Adam optimizer
- CPU-only execution

**Note:**
With very small datasets and long training, the model may memorize captions. This behavior is expected and serves as a demonstration of overfitting in low-data regimes.

## What This Project Is (and Is Not)
### This project is:
- A conceptual and implementation-level exploration of video understanding
- A clean reference for transformer-based video-language pipelines
- Suitable for learning, experimentation, and extension

### This project is not:
- A production-ready or scalable VLM
- A pretrained or large-scale model
- Intended for benchmark comparisons

## Possible Improvements and Future Work

This repository is intentionally minimal. Several improvements can be explored:

### Architectural Improvements
- Replace CNN + temporal encoder with a pure Vision Transformer
- Use a proper Transformer Decoder instead of encoder layers
- Add learnable query tokens in the Q-Former
- Introduce multi-layer Q-Former blocks

### Training Improvements
- Use larger and more diverse video-caption datasets
- Introduce dropout and weight decay
- Add scheduled sampling to reduce exposure bias
- Train with mixed precision or GPUs for scalability

### Generation Improvements
- Add top-p (nucleus) sampling
- Implement beam search
- Condition generation on prompt tokens (instruction-based captioning)

### Evaluation
- Add BLEU / CIDEr metrics
- Evaluate on unseen videos
- Visualize attention maps for interpretability

## Why This Repository Matters

Understanding video-language models requires more than reading papers. This project:
- Forces interaction with every architectural component
- Clarifies how ViTs differ from CNNs in temporal modeling
- Demonstrates how cross-attention enables multimodal reasoning
- Builds intuition for why large VLMs are hard to train from scratch

It is an ideal stepping stone before working with large pretrained models such as **VideoLLaMA**, **Flamingo**, or **GPT-4V**.
