#!/usr/bin/env python3
"""
Training Preparation for Turkish Kokoro-82M Fine-tuning
=======================================================

Two subcommands:

  convert-weights   Download Kokoro-82M weights from HuggingFace and convert
                    them into StyleTTS2's checkpoint format.

  verify            Validate that all training artefacts (train/val lists,
                    audio files, base weights, phoneme vocab) are consistent
                    and ready for training.

Usage:
  uv run python scripts/prepare_training.py convert-weights [--force]
  uv run python scripts/prepare_training.py verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
DATASET_DIR = PROJECT_ROOT / "dataset"
WAVS_DIR = DATASET_DIR / "audio"
TRAIN_LIST = TRAINING_DIR / "train_list.txt"
VAL_LIST = TRAINING_DIR / "val_list.txt"
BASE_WEIGHTS = TRAINING_DIR / "kokoro_base.pth"
CONFIG_PATH = TRAINING_DIR / "config.json"

# HuggingFace source
HF_REPO = "hexgrad/Kokoro-82M"
HF_WEIGHTS_FILE = "kokoro-v1_0.pth"
HF_CONFIG_FILE = "config.json"

# Kokoro-82M components that map to StyleTTS2 model keys
KOKORO_COMPONENTS = [
    "bert",
    "bert_encoder",
    "predictor",
    "text_encoder",
    "decoder",
]


# ===================================================================
# convert-weights
# ===================================================================


def strip_module_prefix(state_dict: dict) -> dict:
    """Remove ``module.`` prefix from all keys (added by DataParallel)."""
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


def convert_weights(force: bool = False) -> None:
    """Download Kokoro-82M weights and repackage for StyleTTS2."""
    if BASE_WEIGHTS.exists() and not force:
        print(f"[Skip] {BASE_WEIGHTS} already exists. Use --force to regenerate.")
        return

    # Late import so the top-level --help stays fast
    import torch
    from huggingface_hub import hf_hub_download

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Download weights from HuggingFace
    # ------------------------------------------------------------------
    print(f"[Download] Fetching {HF_WEIGHTS_FILE} from {HF_REPO} ...")
    weights_path = hf_hub_download(repo_id=HF_REPO, filename=HF_WEIGHTS_FILE)
    print(f"[Download] Cached at {weights_path}")

    print(f"[Download] Fetching {HF_CONFIG_FILE} from {HF_REPO} ...")
    config_path = hf_hub_download(repo_id=HF_REPO, filename=HF_CONFIG_FILE)
    print(f"[Download] Cached at {config_path}")

    # ------------------------------------------------------------------
    # 2. Load the raw checkpoint
    # ------------------------------------------------------------------
    print("[Load] Loading Kokoro-82M weights ...")
    raw = torch.load(weights_path, map_location="cpu", weights_only=True)

    # ------------------------------------------------------------------
    # 3. Strip module. prefix and wrap in StyleTTS2 format
    # ------------------------------------------------------------------
    net: dict[str, dict] = {}
    total_params = 0
    print()
    print(f"{'Component':<20s} {'Parameters':>12s}")
    print("-" * 34)

    for component in sorted(raw.keys()):
        sd = raw[component]
        sd = strip_module_prefix(sd)
        net[component] = sd
        n_params = sum(p.numel() for p in sd.values())
        total_params += n_params
        print(f"{component:<20s} {n_params:>12,d}")

    print("-" * 34)
    print(f"{'TOTAL':<20s} {total_params:>12,d}")
    print()

    state = {"net": net}

    # ------------------------------------------------------------------
    # 4. Save converted weights
    # ------------------------------------------------------------------
    torch.save(state, BASE_WEIGHTS)
    print(f"[Save] Wrote {BASE_WEIGHTS}  ({BASE_WEIGHTS.stat().st_size / 1e6:.1f} MB)")

    # ------------------------------------------------------------------
    # 5. Copy config.json
    # ------------------------------------------------------------------
    import shutil

    shutil.copy2(config_path, CONFIG_PATH)
    print(f"[Save] Copied config to {CONFIG_PATH}")

    print("\nWeight conversion complete.")


# ===================================================================
# verify
# ===================================================================


def _load_list(path: Path) -> list[str]:
    """Read a file and return non-empty lines."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def verify() -> None:
    """Validate training artefacts and report findings."""
    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = []

    # ------------------------------------------------------------------
    # 1. Check train/val list existence
    # ------------------------------------------------------------------
    for label, path in [("train_list.txt", TRAIN_LIST), ("val_list.txt", VAL_LIST)]:
        if not path.exists():
            errors.append(f"{label} not found at {path}")

    if errors:
        # Can't proceed without the lists
        _print_report(errors, warnings, info)
        return

    # ------------------------------------------------------------------
    # 2. Validate format (pipe-separated: filename|phonemes|speaker)
    # ------------------------------------------------------------------
    all_speakers: set[str] = set()
    missing_wavs: list[str] = []
    short_phonemes: list[str] = []
    bad_format: list[str] = []
    train_phonemes: list[str] = []

    for label, path in [("train", TRAIN_LIST), ("val", VAL_LIST)]:
        lines = _load_list(path)
        info.append(f"{label}: {len(lines)} entries in {path.name}")

        for i, line in enumerate(lines, 1):
            parts = line.split("|")
            if len(parts) != 3:
                bad_format.append(f"{label} line {i}: expected 3 fields, got {len(parts)}")
                continue

            filename, phonemes, speaker = parts

            # Speaker tracking
            all_speakers.add(speaker)

            # WAV existence
            wav_path = WAVS_DIR / filename
            if not wav_path.exists():
                missing_wavs.append(f"{label} line {i}: {wav_path}")

            # Empty/short phonemes
            if len(phonemes.strip()) < 3:
                short_phonemes.append(
                    f"{label} line {i}: phonemes too short ({phonemes!r})"
                )

            # Collect train phonemes for vocab check (first 50 only)
            if label == "train" and len(train_phonemes) < 50:
                train_phonemes.append(phonemes)

    if bad_format:
        for msg in bad_format[:10]:
            errors.append(f"Bad format: {msg}")
        if len(bad_format) > 10:
            errors.append(f"... and {len(bad_format) - 10} more format errors")

    if missing_wavs:
        for msg in missing_wavs[:10]:
            errors.append(f"Missing WAV: {msg}")
        if len(missing_wavs) > 10:
            errors.append(f"... and {len(missing_wavs) - 10} more missing WAVs")

    if short_phonemes:
        for msg in short_phonemes[:10]:
            warnings.append(f"Short phonemes: {msg}")
        if len(short_phonemes) > 10:
            warnings.append(
                f"... and {len(short_phonemes) - 10} more short-phoneme entries"
            )

    # ------------------------------------------------------------------
    # 3. List speakers
    # ------------------------------------------------------------------
    info.append(f"Speakers found: {sorted(all_speakers)}")

    # ------------------------------------------------------------------
    # 4. Verify base weights
    # ------------------------------------------------------------------
    if BASE_WEIGHTS.exists():
        try:
            import torch

            state = torch.load(BASE_WEIGHTS, map_location="cpu", weights_only=True)
            if "net" not in state:
                errors.append(
                    f"kokoro_base.pth missing 'net' key (found: {list(state.keys())})"
                )
            else:
                components = sorted(state["net"].keys())
                total = sum(
                    sum(p.numel() for p in sd.values())
                    for sd in state["net"].values()
                )
                info.append(
                    f"kokoro_base.pth: components={components}, "
                    f"total params={total:,d}"
                )
        except Exception as exc:
            errors.append(f"Failed to load kokoro_base.pth: {exc}")
    else:
        warnings.append(
            f"kokoro_base.pth not found at {BASE_WEIGHTS}. "
            "Run 'convert-weights' first."
        )

    # ------------------------------------------------------------------
    # 5. Phoneme vocab check (first 50 train entries)
    # ------------------------------------------------------------------
    if CONFIG_PATH.exists() and train_phonemes:
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
            vocab = set(config.get("vocab", {}).keys())
            if not vocab:
                warnings.append("config.json has no 'vocab' field; skipping vocab check")
            else:
                unknown_chars: dict[str, int] = {}
                for ph in train_phonemes:
                    for ch in ph:
                        if ch not in vocab:
                            unknown_chars[ch] = unknown_chars.get(ch, 0) + 1

                if unknown_chars:
                    warnings.append(
                        f"Found {len(unknown_chars)} phoneme symbol(s) not in Kokoro vocab "
                        f"(checked first {len(train_phonemes)} train entries):"
                    )
                    for ch in sorted(unknown_chars, key=lambda c: -unknown_chars[c]):
                        warnings.append(
                            f"  U+{ord(ch):04X} {ch!r:6s} x{unknown_chars[ch]}"
                        )
                else:
                    info.append(
                        f"Vocab check passed: all symbols in first "
                        f"{len(train_phonemes)} train entries are in Kokoro vocab"
                    )
        except Exception as exc:
            warnings.append(f"Could not verify vocab from config.json: {exc}")
    elif not CONFIG_PATH.exists():
        warnings.append(
            f"config.json not found at {CONFIG_PATH}; skipping vocab check. "
            "Run 'convert-weights' first."
        )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    _print_report(errors, warnings, info)


def _print_report(
    errors: list[str], warnings: list[str], info: list[str]
) -> None:
    """Print a structured verification report."""
    print()
    print("=" * 60)
    print("TRAINING VERIFICATION REPORT")
    print("=" * 60)

    if info:
        print("\n--- Info ---")
        for msg in info:
            print(f"  [OK] {msg}")

    if warnings:
        print("\n--- Warnings ---")
        for msg in warnings:
            print(f"  [WARN] {msg}")

    if errors:
        print("\n--- Errors ---")
        for msg in errors:
            print(f"  [ERR] {msg}")

    print()
    if errors:
        print(f"RESULT: FAIL ({len(errors)} error(s), {len(warnings)} warning(s))")
    elif warnings:
        print(f"RESULT: OK with warnings ({len(warnings)} warning(s))")
    else:
        print("RESULT: ALL CHECKS PASSED")
    print("=" * 60)
    print()

    if errors:
        sys.exit(1)


# ===================================================================
# CLI
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Training preparation for Turkish Kokoro-82M fine-tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Convert Kokoro-82M weights to StyleTTS2 format
  uv run python scripts/prepare_training.py convert-weights

  # Force re-download and re-convert
  uv run python scripts/prepare_training.py convert-weights --force

  # Verify all training artefacts are ready
  uv run python scripts/prepare_training.py verify
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # -- convert-weights --
    cw_parser = subparsers.add_parser(
        "convert-weights",
        help="Download Kokoro-82M weights and convert to StyleTTS2 format.",
    )
    cw_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-convert even if output already exists.",
    )

    # -- verify --
    subparsers.add_parser(
        "verify",
        help="Validate train/val lists, audio files, weights, and phoneme vocab.",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "convert-weights":
        convert_weights(force=args.force)
    elif args.command == "verify":
        verify()


if __name__ == "__main__":
    main()
