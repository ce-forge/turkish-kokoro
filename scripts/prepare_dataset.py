#!/usr/bin/env python3
"""
Prepare Turkish TTS Dataset for Kokoro-82M Fine-tuning
======================================================

Downloads the Mazlum Kiper Turkish TTS subset from HuggingFace, resamples
audio to 24 kHz mono, runs Turkish G2P (via misaki/espeak), performs a
phoneme audit against the Kokoro-82M vocabulary, and writes StyleTTS2-format
train/val list files.

Modes:
  default   -- full pipeline: download, resample, G2P, audit, split, write
  --audit-only  -- skip download; re-run G2P + audit on existing metadata

Output layout:
  dataset/audio/tm_mazlum/NNNNNN.wav   WAV files (24 kHz, mono, 16-bit)
  dataset/metadata.jsonl                per-clip metadata (for resumability)
  dataset/stats.json                    summary statistics
  training/train_list.txt               StyleTTS2 train list
  training/val_list.txt                 StyleTTS2 val list
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
AUDIO_DIR = DATASET_DIR / "audio"
METADATA_PATH = DATASET_DIR / "metadata.jsonl"
STATS_PATH = DATASET_DIR / "stats.json"
TRAINING_DIR = PROJECT_ROOT / "training"
TRAIN_LIST_PATH = TRAINING_DIR / "train_list.txt"
VAL_LIST_PATH = TRAINING_DIR / "val_list.txt"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPEAKER_SOURCE_REPO = "omersaidd/tts_mazlum_kiper_tur"
SPEAKER_NAME = "tm_mazlum"
TARGET_SR = 24000
MIN_DURATION_S = 2.0
MAX_DURATION_S = 30.0
MIN_WORDS = 3
VAL_RATIO = 0.05
RANDOM_SEED = 42

# Phoneme fixups: IPA symbols produced by espeak-tr that are not in the
# Kokoro-82M vocabulary.  dark L (ɫ) -> plain l.
PHONEME_FIXUPS: dict[str, str] = {"ɫ": "l"}


# ---------------------------------------------------------------------------
# G2P helpers
# ---------------------------------------------------------------------------
def get_g2p():
    """Return a Turkish grapheme-to-phoneme function.

    Tries misaki (espeak backend) first, falls back to phonemizer.
    """
    try:
        from misaki import espeak

        g2p_obj = espeak.EspeakG2P(language="tr")

        def g2p(text: str) -> str:
            ph, _ = g2p_obj(text)
            for old, new in PHONEME_FIXUPS.items():
                ph = ph.replace(old, new)
            return ph

        g2p("Merhaba")  # smoke test
        print("[G2P] Using misaki EspeakG2P (language=tr)")
        return g2p
    except Exception as exc:
        print(f"[G2P] misaki unavailable ({exc}), falling back to phonemizer")
        from phonemizer import phonemize

        def g2p(text: str) -> str:
            ph = phonemize(text, language="tr", backend="espeak", strip=True)
            for old, new in PHONEME_FIXUPS.items():
                ph = ph.replace(old, new)
            return ph.strip()

        g2p("Merhaba")  # smoke test
        print("[G2P] Using phonemizer (language=tr, backend=espeak)")
        return g2p


# ---------------------------------------------------------------------------
# Phoneme audit
# ---------------------------------------------------------------------------
def run_phoneme_audit(all_phonemes: list[str]) -> dict:
    """Compare unique IPA characters against the Kokoro-82M vocab.

    Returns a dict with ``covered``, ``unmapped``, and ``unmapped_counts``.
    """
    # Import Kokoro symbol table
    sys.path.insert(0, str(TRAINING_DIR))
    from kokoro_symbols import dicts as kokoro_vocab

    # Collect character-level statistics
    char_counter: Counter[str] = Counter()
    for ph in all_phonemes:
        for ch in ph:
            char_counter[ch] += 1

    all_ipa_chars = sorted(char_counter.keys())

    covered: set[str] = set()
    unmapped: set[str] = set()
    for ch in all_ipa_chars:
        if ch in kokoro_vocab:
            covered.add(ch)
        else:
            unmapped.add(ch)

    # Print report
    print("\n" + "=" * 60)
    print("PHONEME AUDIT")
    print("=" * 60)
    print(f"Unique IPA characters: {len(all_ipa_chars)}")
    print(f"Covered by Kokoro vocab: {len(covered)}")
    print(f"Unmapped (NOT in Kokoro vocab): {len(unmapped)}")

    if unmapped:
        print("\nUnmapped symbols (add to PHONEME_FIXUPS or extend vocab):")
        for ch in sorted(unmapped):
            print(
                f"  U+{ord(ch):04X}  {ch!r:6s}  "
                f"occurrences: {char_counter[ch]}"
            )
    else:
        print("\nAll symbols covered -- no fixups needed.")

    print("=" * 60 + "\n")

    unmapped_counts = {ch: char_counter[ch] for ch in sorted(unmapped)}
    return {
        "covered": sorted(covered),
        "unmapped": sorted(unmapped),
        "unmapped_counts": unmapped_counts,
    }


# ---------------------------------------------------------------------------
# Download & resample
# ---------------------------------------------------------------------------
def download_and_resample(max_clips: int | None = None) -> list[dict]:
    """Stream the HuggingFace dataset, filter, resample, and save WAVs.

    Returns a list of metadata dicts (one per accepted clip).
    Already-processed clips (detected via metadata.jsonl) are skipped
    for resumability.
    """
    from datasets import load_dataset

    speaker_dir = AUDIO_DIR / SPEAKER_NAME
    speaker_dir.mkdir(parents=True, exist_ok=True)

    # Load existing metadata for resumability
    existing: dict[str, dict] = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                existing[entry["id"]] = entry
        print(f"[Resume] Found {len(existing)} existing entries in metadata")

    ds = load_dataset(SPEAKER_SOURCE_REPO, split="train", streaming=True)

    entries: list[dict] = list(existing.values())
    clip_idx = len(existing)
    skipped_dur = 0
    skipped_words = 0
    processed = 0

    meta_fp = open(METADATA_PATH, "a", encoding="utf-8")

    try:
        for sample in tqdm(ds, desc="Downloading & resampling", unit="clip"):
            # Determine a unique ID for this sample
            sample_id = str(sample.get("id", processed))
            processed += 1

            if sample_id in existing:
                continue

            # Extract text
            text = sample.get("text", sample.get("sentence", "")).strip()
            if not text:
                continue
            if len(text.split()) < MIN_WORDS:
                skipped_words += 1
                continue

            # Extract audio
            audio_data = sample["audio"]
            source_sr = audio_data["sampling_rate"]
            waveform = audio_data["array"]  # numpy float array

            # Duration filter
            duration = len(waveform) / source_sr
            if duration < MIN_DURATION_S or duration > MAX_DURATION_S:
                skipped_dur += 1
                continue

            # Convert to torch tensor and resample
            wav_tensor = torch.from_numpy(waveform).float()
            if wav_tensor.ndim == 1:
                wav_tensor = wav_tensor.unsqueeze(0)  # (1, T)
            elif wav_tensor.shape[0] > wav_tensor.shape[1]:
                # Likely (T, C) -- transpose
                wav_tensor = wav_tensor.T
            # Take first channel (mono)
            wav_tensor = wav_tensor[0:1]

            if source_sr != TARGET_SR:
                wav_tensor = AF.resample(wav_tensor, source_sr, TARGET_SR)

            # Normalize to 16-bit range
            wav_np = wav_tensor.squeeze(0).numpy()
            peak = np.abs(wav_np).max()
            if peak > 0:
                wav_np = wav_np / peak * 0.95  # leave headroom

            # Save WAV
            filename = f"{clip_idx:06d}.wav"
            wav_path = speaker_dir / filename
            sf.write(str(wav_path), wav_np, TARGET_SR, subtype="PCM_16")

            new_duration = len(wav_np) / TARGET_SR

            entry = {
                "id": sample_id,
                "filename": f"{SPEAKER_NAME}/{filename}",
                "text": text,
                "duration": round(new_duration, 3),
                "source_sr": source_sr,
            }
            entries.append(entry)
            meta_fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
            meta_fp.flush()

            clip_idx += 1

            if max_clips is not None and clip_idx >= max_clips:
                print(f"[Limit] Reached max_clips={max_clips}, stopping.")
                break
    finally:
        meta_fp.close()

    print(f"\n[Download] Total accepted: {len(entries)}")
    print(f"[Download] Skipped (duration): {skipped_dur}")
    print(f"[Download] Skipped (word count): {skipped_words}")

    return entries


# ---------------------------------------------------------------------------
# Load existing metadata
# ---------------------------------------------------------------------------
def load_metadata() -> list[dict]:
    """Read dataset/metadata.jsonl and return list of dicts."""
    if not METADATA_PATH.exists():
        print(f"[Error] {METADATA_PATH} not found. Run download first.")
        sys.exit(1)

    entries: list[dict] = []
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"[Metadata] Loaded {len(entries)} entries from {METADATA_PATH}")
    return entries


# ---------------------------------------------------------------------------
# G2P pass
# ---------------------------------------------------------------------------
def run_g2p_pass(entries: list[dict]) -> list[dict]:
    """Run G2P on all entries, store result in entry['phonemes']."""
    g2p = get_g2p()

    failed = 0
    for entry in tqdm(entries, desc="Running G2P", unit="clip"):
        try:
            ph = g2p(entry["text"])
            entry["phonemes"] = ph
        except Exception as exc:
            print(f"[G2P] Failed on '{entry['text'][:60]}...': {exc}")
            entry["phonemes"] = ""
            failed += 1

    print(f"[G2P] Completed. Failures: {failed}/{len(entries)}")
    return entries


# ---------------------------------------------------------------------------
# Filter & split
# ---------------------------------------------------------------------------
MIN_PHONEME_LEN = 3  # minimum number of phoneme characters


def filter_and_split(entries: list[dict]) -> tuple[list[dict], list[dict]]:
    """Filter entries with empty/short phonemes, then train/val split."""
    valid = [e for e in entries if len(e.get("phonemes", "")) >= MIN_PHONEME_LEN]
    dropped = len(entries) - len(valid)
    if dropped:
        print(f"[Filter] Dropped {dropped} entries with empty/short phonemes")

    random.seed(RANDOM_SEED)
    random.shuffle(valid)

    val_count = max(1, int(len(valid) * VAL_RATIO))
    val_entries = valid[:val_count]
    train_entries = valid[val_count:]

    print(f"[Split] Train: {len(train_entries)}, Val: {len(val_entries)}")
    return train_entries, val_entries


# ---------------------------------------------------------------------------
# Write StyleTTS2-format lists
# ---------------------------------------------------------------------------
def write_lists(
    train_entries: list[dict],
    val_entries: list[dict],
) -> None:
    """Write training/train_list.txt and training/val_list.txt.

    Format per line:  <relative_wav_path>|<phonemes>|<speaker_id>
    StyleTTS2's meldataset.py expects an integer speaker ID.
    """
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    def _write(path: Path, entries: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for e in entries:
                line = f"{e['filename']}|{e['phonemes']}|0"
                f.write(line + "\n")
        print(f"[Write] {path}  ({len(entries)} lines)")

    _write(TRAIN_LIST_PATH, train_entries)
    _write(VAL_LIST_PATH, val_entries)


# ---------------------------------------------------------------------------
# Write stats
# ---------------------------------------------------------------------------
def write_stats(
    entries: list[dict],
    train_entries: list[dict],
    val_entries: list[dict],
    audit_result: dict,
) -> None:
    """Write dataset/stats.json with summary numbers."""
    durations = [e["duration"] for e in entries if "duration" in e]

    stats = {
        "speaker": SPEAKER_NAME,
        "source_repo": SPEAKER_SOURCE_REPO,
        "total_clips": len(entries),
        "train_clips": len(train_entries),
        "val_clips": len(val_entries),
        "total_duration_s": round(sum(durations), 1),
        "mean_duration_s": round(sum(durations) / max(len(durations), 1), 2),
        "min_duration_s": round(min(durations), 2) if durations else 0,
        "max_duration_s": round(max(durations), 2) if durations else 0,
        "target_sr": TARGET_SR,
        "phoneme_fixups": PHONEME_FIXUPS,
        "audit_covered_count": len(audit_result.get("covered", [])),
        "audit_unmapped_count": len(audit_result.get("unmapped", [])),
        "audit_unmapped": audit_result.get("unmapped", []),
    }

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[Stats] Written to {STATS_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Turkish TTS dataset for Kokoro-82M fine-tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Full pipeline (download + G2P + audit + split)
  python scripts/prepare_dataset.py

  # Download at most 100 clips (for testing)
  python scripts/prepare_dataset.py --max-clips 100

  # Re-run only G2P + audit on existing metadata
  python scripts/prepare_dataset.py --audit-only
""",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Skip download; re-run G2P and phoneme audit on existing metadata.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Maximum number of clips to download (for testing).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1: Obtain entries
    # ------------------------------------------------------------------
    if args.audit_only:
        print("=== AUDIT-ONLY MODE ===")
        entries = load_metadata()
    else:
        print("=== FULL PIPELINE ===")
        entries = download_and_resample(max_clips=args.max_clips)

    if not entries:
        print("[Error] No entries to process. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: G2P
    # ------------------------------------------------------------------
    entries = run_g2p_pass(entries)

    # ------------------------------------------------------------------
    # Step 3: Phoneme audit
    # ------------------------------------------------------------------
    all_phonemes = [e["phonemes"] for e in entries if e.get("phonemes")]
    audit_result = run_phoneme_audit(all_phonemes)

    # ------------------------------------------------------------------
    # Step 4: Filter & split
    # ------------------------------------------------------------------
    train_entries, val_entries = filter_and_split(entries)

    # ------------------------------------------------------------------
    # Step 5: Write lists & stats
    # ------------------------------------------------------------------
    write_lists(train_entries, val_entries)
    write_stats(entries, train_entries, val_entries, audit_result)

    print("\nDone.")


if __name__ == "__main__":
    main()
