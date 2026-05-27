# Turkish Kokoro

Training Kokoro-82M (StyleTTS 2 architecture) for Turkish TTS, forked from the [kikiri-tts](https://github.com/semidark/kikiri-tts) German training pipeline.

## Project Context

This project produces a Turkish voice model for the [Pryzm](../pryzm/) assistant's Kokoro TTS sidecar. It is a separate repo because training and inference have different dependency trees.

Research notes: `../pryzm/docs/internal/2026-05-27-turkish-kokoro-training-research.md`

## Architecture

- **Model**: Kokoro-82M (82M params, ISTFTNet vocoder, no diffusion)
- **Training framework**: Patched StyleTTS2 (semidark fork) via `accelerate`
- **G2P**: espeak-ng Turkish → Kokoro 178-token symbol set
- **Dataset**: afkfatih/turkish-tts-combined-raw (HuggingFace)

## Environment

- Python managed by `uv` (pyproject.toml)
- GPU: RTX 5070 Ti (16GB VRAM)
- System deps: `espeak-ng`, `libsndfile1`

## Commands

```bash
# Environment
uv sync

# Dataset preparation
uv run python scripts/prepare_dataset.py download
uv run python scripts/prepare_dataset.py format

# Training preparation
uv run python scripts/prepare_training.py prepare
uv run python scripts/prepare_training.py convert-weights
uv run python scripts/prepare_training.py verify

# Training (run from StyleTTS2/)
accelerate launch train_first.py --config_path ../configs/config_turkish_ft.yml
accelerate launch train_second.py --config_path ../configs/config_turkish_ft.yml

# Voicepack extraction & inference test
uv run python scripts/extract_voicepack.py --model <checkpoint> --audio-dir <dir> --output <out.pt>
uv run python scripts/test_inference.py --checkpoint <checkpoint> --voicepack <voice.pt>
```

## Key Files

- `configs/config_turkish_ft.yml` — Training hyperparameters
- `training/kokoro_symbols.py` — Kokoro 178-token vocab mapping (shared with kikiri-tts)
- `scripts/prepare_dataset.py` — Download and format afkfatih dataset
- `scripts/prepare_training.py` — Train/val split, weight conversion, verification
- `scripts/extract_voicepack.py` — Extract voicepack from trained checkpoint
- `scripts/test_inference.py` — Turkish test sentence inference

## Conventions

- Follow kikiri-tts patterns for training scripts and config layout
- StyleTTS2 config parameters go at top level, not nested under `training:`
- Phoneme fixups for unmapped IPA symbols go in a `PHONEME_FIXUPS` dict
- Speaker names follow Kokoro convention: `tf_name` (female), `tm_name` (male)
