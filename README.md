# Kaka TTS

> A specification and design for a Text-to-Speech model that speaks authentic **Telangana slang**.

## Overview

Kaka Project is the planning home for **Kaka TTS**, a specialized Text-to-Speech system aimed at generating natural, culturally-authentic Telangana slang speech — a regional Telugu dialect underserved by current voice assistants. At this stage the repository holds the **requirements** and **technical design** for the system, not a running implementation. The design centers on fine-tuning a pre-trained VITS model on Telangana slang audio, augmented with a custom "Cultural Context Layer" that captures regional phonemes and prosody.

The documents here are the source of truth for what the system should do and how it is intended to be built.

## Features

This repo currently contains detailed design and requirements artifacts covering:

- **Audio preprocessing spec** — normalization, silence removal, segmentation (5–15s), and mel-spectrogram feature extraction
- **Text & phoneme processing spec** — Telangana-specific phoneme mapping, Telugu script handling, and code-mixed (Telugu/English) text rules
- **Model architecture** — a VITS base model plus an attention/LSTM-based Cultural Context Layer (~40M params)
- **Training pipeline design** — transfer learning, mixed-precision (FP16), gradient accumulation, checkpointing, and early stopping
- **Inference & serving design** — a FastAPI REST endpoint returning WAV/MP3, with batching and request queuing
- **Evaluation plan** — objective metrics (MOS/PESQ/STOI), WER via ASR, prosody comparison, and a regional-authenticity metric
- **Infrastructure plan** — AWS g5.2xlarge spot instances under a ~$1,200 training budget
- **Correctness properties** — formal, verifiable statements tied to specific requirements

## Tech Stack

The design targets the following stack (planned, per `design.md`):

- **Language:** Python
- **ML framework:** PyTorch (with AMP mixed precision, DistributedDataParallel)
- **TTS model:** VITS (end-to-end) with a HiFi-GAN vocoder; Coqui TTS referenced as a base
- **Audio:** librosa
- **Serving:** FastAPI
- **Evaluation:** NISQA, PESQ, pystoi, Whisper / IndicWav2Vec (ASR for Telugu)
- **Infrastructure:** AWS (g5.2xlarge, S3), YAML-based configuration

## Getting Started

This repository currently contains specification documents rather than executable code. To review the project:

```bash
git clone https://github.com/nickthelegend/kaka-project.git
cd kaka-project

# Read the specs
open requirements.md   # what the system must do (EARS-style acceptance criteria)
open design.md         # architecture, components, and interfaces
```

The specs were authored as [Kiro](https://kiro.dev) specs and also live under `.kiro/specs/kaka-tts/`.

## Project Structure

```
kaka-project/
├── requirements.md              # Requirements & acceptance criteria for Kaka TTS
├── design.md                    # Architecture, component interfaces, data models
├── .kiro/
│   └── specs/
│       └── kaka-tts/            # Kiro spec source (requirements.md, design.md)
└── .vscode/
    └── settings.json
```

---

Built by [**nickthelegend**](https://github.com/nickthelegend) · [nickthelegend.tech](https://nickthelegend.tech)
