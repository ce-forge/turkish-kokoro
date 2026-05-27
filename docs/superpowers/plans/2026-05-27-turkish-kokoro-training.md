# Turkish Kokoro Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete training pipeline that takes Mazlum Kiper's Turkish audio clips and produces a Kokoro-compatible Turkish TTS model.

**Architecture:** Fork kikiri-tts's patched StyleTTS2 pipeline, replace German dataset/G2P with Turkish equivalents. Initialize from English Kokoro-82M weights, train Stage 1 (acoustic) then Stage 2 (prosody) on a single speaker. Extract voicepack and test inference.

**Tech Stack:** Python 3.12 (uv), PyTorch + CUDA 12.8, StyleTTS2 (semidark fork), espeak-ng, torchaudio, HuggingFace datasets

---

### Task 1: Add Git Submodules and System Dependencies

**Files:**
- Create: `.gitmodules`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add StyleTTS2 and kokoro submodules**

```bash
git submodule add https://github.com/semidark/StyleTTS2.git StyleTTS2
git submodule add https://github.com/semidark/kokoro.git kokoro
```

- [ ] **Step 2: Verify submodules cloned correctly**

```bash
ls StyleTTS2/train_first.py StyleTTS2/train_second.py StyleTTS2/models.py
ls kokoro/kokoro/__init__.py
```

Expected: all files exist.

- [ ] **Step 3: Install system dependencies**

```bash
sudo apt-get install -y espeak-ng libsndfile1
```

- [ ] **Step 4: Verify espeak-ng Turkish support**

```bash
espeak-ng --voices=tr
```

Expected: output includes a line with `tr` language code.

- [ ] **Step 5: Update pyproject.toml with full dependencies**

```toml
[project]
name = "turkish-kokoro"
version = "0.1.0"
description = "Training Kokoro-82M for Turkish TTS using StyleTTS2"
readme = "README.md"
license = { text = "Apache-2.0" }
requires-python = ">=3.10, <3.14"
dependencies = [
    "datasets",
    "huggingface_hub",
    "loguru",
    "misaki[en]>=0.9.4",
    "numpy",
    "soundfile",
    "torch",
    "torchaudio",
    "torchcodec",
    "tqdm",
    "transformers",
]

[tool.uv]
find-links = ["https://download.pytorch.org/whl/cu128"]

[tool.uv.sources]
misaki = { git = "https://github.com/semidark/misaki.git", branch = "main" }
```

- [ ] **Step 6: Sync environment**

```bash
uv sync
```

Expected: all packages install without errors.

- [ ] **Step 7: Verify misaki espeak Turkish G2P works**

```bash
uv run python3 -c "
from misaki import espeak
g2p = espeak.EspeakG2P(language='tr')
phonemes, _ = g2p('Merhaba dünya')
print(f'Input: Merhaba dünya')
print(f'Phonemes: {phonemes}')
assert len(phonemes) > 0, 'G2P produced empty output'
print('OK')
"
```

Expected: prints IPA phonemes for "Merhaba dünya". If misaki's `EspeakG2P` doesn't accept `language='tr'`, fall back to using the `phonemizer` package directly:

```bash
uv pip install phonemizer
uv run python3 -c "
from phonemizer import phonemize
result = phonemize('Merhaba dünya', language='tr', backend='espeak')
print(f'Phonemes: {result}')
assert len(result) > 0
print('OK')
"
```

If falling back, update all subsequent G2P references in this plan to use `phonemizer` instead of `misaki.espeak`.

- [ ] **Step 8: Commit**

```bash
git add .gitmodules StyleTTS2 kokoro pyproject.toml uv.lock
git commit -m "Add StyleTTS2 and kokoro submodules, update dependencies"
```

---

### Task 2: Copy Kokoro Symbol Map and Verify

**Files:**
- Copy: `training/kokoro_symbols.py` (from kikiri-tts repo at `/tmp/kikiri-tts/training/kokoro_symbols.py`)
- Copy: `training/OOD_texts.txt` (create new with Turkish content)

- [ ] **Step 1: Copy kokoro_symbols.py from kikiri-tts**

```bash
cp /tmp/kikiri-tts/training/kokoro_symbols.py training/kokoro_symbols.py
```

If `/tmp/kikiri-tts` no longer exists:

```bash
git clone --depth 1 https://github.com/semidark/kikiri-tts.git /tmp/kikiri-tts
cp /tmp/kikiri-tts/training/kokoro_symbols.py training/kokoro_symbols.py
```

- [ ] **Step 2: Verify symbol map integrity**

```bash
uv run python3 -c "
import sys
sys.path.insert(0, 'training')
from kokoro_symbols import symbols, dicts, TextCleaner
assert len(symbols) == 178, f'Expected 178 symbols, got {len(symbols)}'
assert dicts['ʃ'] == 131, 'ş mapping wrong'
assert dicts['ʤ'] == 82, 'c mapping wrong'
assert dicts['ʧ'] == 133, 'ç mapping wrong'
assert dicts['ɯ'] == 110, 'ı mapping wrong'
assert dicts['ø'] == 116, 'ö mapping wrong'
assert dicts['y'] == 67, 'ü mapping wrong'
assert dicts['ɾ'] == 125, 'r mapping wrong'
assert dicts['ˈ'] == 156, 'stress mapping wrong'
assert dicts['ː'] == 158, 'length mapping wrong'
tc = TextCleaner()
result = tc('ˈmɛɾˌhɑbɑ')
assert len(result) > 0, 'TextCleaner produced empty output'
print(f'All Turkish phoneme mappings verified. TextCleaner output: {result}')
print('OK')
"
```

- [ ] **Step 3: Create Turkish OOD texts**

Create `training/OOD_texts.txt` with ~20 Turkish sentences converted to IPA. This file is used during training for out-of-domain regularization.

```bash
uv run python3 -c "
from misaki import espeak

g2p = espeak.EspeakG2P(language='tr')

sentences = [
    'Bugün hava çok güzel, dışarı çıkmak istiyorum.',
    'Türkiye\'nin başkenti Ankara\'dır ve en büyük şehri İstanbul\'dur.',
    'Bilim insanları yeni bir gezegen keşfetti.',
    'Lütfen kapıyı kapatır mısınız, dışarısı çok soğuk.',
    'Deprem bölgesine yardım malzemeleri gönderildi.',
    'Bu kitabı okumak için sabırsızlanıyorum.',
    'Öğretmen öğrencilere yeni konuyu anlattı.',
    'Akdeniz kıyısında güneşli bir tatil köyü var.',
    'Müzisyenler konser için sahneye çıktı.',
    'Çocuklar parkta neşeyle koşuşturuyordu.',
    'Ekonomideki değişiklikler herkesi etkiledi.',
    'Doktor hastaya ilaçlarını düzenli almasını söyledi.',
    'Üniversite sınavına hazırlanan öğrenciler çok çalışıyor.',
    'Yeni açılan restoranda deniz ürünleri çok lezzetli.',
    'Futbol maçını izlemek için stadyuma gittik.',
    'Kütüphanede sessizce oturup çalışmak güzel.',
    'Mühendisler köprünün inşaatını tamamladı.',
    'Bahar gelince çiçekler açmaya başladı.',
    'Bu sorunun çözümü için birlikte düşünmeliyiz.',
    'Geçen hafta sonu ailecek pikniğe gittik.',
]

phonemes_list = []
for text in sentences:
    ph, _ = g2p(text)
    phonemes_list.append(ph)
    print(f'{text[:50]:50s} -> {ph[:50]}...')

with open('training/OOD_texts.txt', 'w') as f:
    f.write('\n'.join(phonemes_list) + '\n')

print(f'\nWrote {len(phonemes_list)} OOD phoneme sequences to training/OOD_texts.txt')
"
```

If misaki doesn't work for Turkish, use the phonemizer fallback:

```bash
uv run python3 -c "
from phonemizer import phonemize

sentences = [...]  # same list as above

phonemes_list = []
for text in sentences:
    ph = phonemize(text, language='tr', backend='espeak', strip=True)
    phonemes_list.append(ph.strip())

with open('training/OOD_texts.txt', 'w') as f:
    f.write('\n'.join(phonemes_list) + '\n')
"
```

- [ ] **Step 4: Commit**

```bash
git add training/kokoro_symbols.py training/OOD_texts.txt
git commit -m "Add Kokoro symbol map and Turkish OOD texts"
```

---

### Task 3: Write Dataset Preparation Script

**Files:**
- Create: `scripts/prepare_dataset.py`

This script downloads the Mazlum Kiper subset from HuggingFace, resamples to 24kHz, runs Turkish G2P, performs a phoneme audit, and writes StyleTTS2-format train/val lists.

- [ ] **Step 1: Write prepare_dataset.py**

```python
#!/usr/bin/env python3
"""
Turkish Kokoro: Dataset Preparation
=====================================
Downloads Mazlum Kiper's Turkish speech clips from HuggingFace,
resamples to 24kHz, runs Turkish G2P, and writes StyleTTS2-format
train/val lists.

Usage:
    # Download and prepare everything
    uv run python scripts/prepare_dataset.py

    # Just run phoneme audit (after download)
    uv run python scripts/prepare_dataset.py --audit-only
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torchaudio
import torch
from datasets import load_dataset
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────

DATASET_REPO = "afkfatih/turkish-tts-combined-raw"
SPEAKER_SOURCE_REPO = "omersaidd/tts_mazlum_kiper_tur"
SPEAKER_NAME = "tm_mazlum"

DATASET_DIR = Path("./dataset")
AUDIO_DIR = DATASET_DIR / "audio" / SPEAKER_NAME
TRAINING_DIR = Path("./training")
TRAIN_LIST = TRAINING_DIR / "train_list.txt"
VAL_LIST = TRAINING_DIR / "val_list.txt"
STATS_FILE = DATASET_DIR / "stats.json"

TARGET_SR = 24000
MIN_DURATION_S = 2.0
MAX_DURATION_S = 30.0
MIN_WORDS = 3
VAL_RATIO = 0.05
RANDOM_SEED = 42

# Turkish phoneme fixups: IPA symbols from espeak-ng that aren't in Kokoro's
# 178-token vocab. Discovered by running the phoneme audit pass.
# Start empty — the audit step will tell us what to add.
PHONEME_FIXUPS: dict[str, str] = {}


def get_g2p():
    """Get a Turkish G2P function. Try misaki first, fall back to phonemizer."""
    try:
        from misaki import espeak
        g2p_obj = espeak.EspeakG2P(language='tr')
        def g2p(text):
            ph, _ = g2p_obj(text)
            for old, new in PHONEME_FIXUPS.items():
                ph = ph.replace(old, new)
            return ph
        # Smoke test
        g2p("Merhaba")
        print("G2P backend: misaki (espeak-ng)")
        return g2p
    except Exception as e:
        print(f"misaki G2P failed ({e}), trying phonemizer...")

    try:
        from phonemizer import phonemize
        def g2p(text):
            ph = phonemize(text, language='tr', backend='espeak', strip=True)
            for old, new in PHONEME_FIXUPS.items():
                ph = ph.replace(old, new)
            return ph.strip()
        g2p("Merhaba")
        print("G2P backend: phonemizer (espeak-ng)")
        return g2p
    except Exception as e:
        print(f"ERROR: No working G2P backend. Install espeak-ng and misaki or phonemizer.")
        sys.exit(1)


def download_and_prepare(audit_only: bool = False):
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    # ── Download ────────────────────────────────────────────────────────
    if audit_only:
        print("Audit-only mode: skipping download, using existing files.")
    else:
        print(f"Loading dataset: {SPEAKER_SOURCE_REPO}")
        ds = load_dataset(SPEAKER_SOURCE_REPO, split="train", streaming=True)

        print(f"Downloading and resampling to {TARGET_SR}Hz...")
        entries = []
        skipped = 0

        for i, sample in enumerate(tqdm(ds, desc="Downloading")):
            audio = sample["audio"]
            text = sample["text"]
            arr = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            duration = len(arr) / sr

            # Filter
            if duration < MIN_DURATION_S or duration > MAX_DURATION_S:
                skipped += 1
                continue
            if len(text.split()) < MIN_WORDS:
                skipped += 1
                continue

            # Resample to 24kHz
            if sr != TARGET_SR:
                waveform = torch.from_numpy(arr).unsqueeze(0)
                waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
                arr = waveform.squeeze(0).numpy()

            # Save WAV
            wav_name = f"{i:06d}.wav"
            wav_path = AUDIO_DIR / wav_name
            sf.write(str(wav_path), arr, TARGET_SR, subtype="PCM_16")

            entries.append({
                "index": i,
                "wav_name": wav_name,
                "text": text,
                "duration": len(arr) / TARGET_SR,
            })

        print(f"\nDownloaded: {len(entries):,} clips | Skipped: {skipped:,}")

        # Save metadata for resumability
        meta_path = DATASET_DIR / "metadata.jsonl"
        with open(meta_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"Wrote {meta_path}")

    # ── Load metadata ───────────────────────────────────────────────────
    meta_path = DATASET_DIR / "metadata.jsonl"
    if not meta_path.exists():
        print("ERROR: No metadata.jsonl found. Run without --audit-only first.")
        sys.exit(1)

    entries = []
    with open(meta_path) as f:
        for line in f:
            entries.append(json.loads(line))
    print(f"\nLoaded {len(entries):,} entries from metadata.")

    # ── G2P ─────────────────────────────────────────────────────────────
    g2p = get_g2p()

    print("Running Turkish G2P on all transcriptions...")
    all_ipa_chars = set()
    g2p_errors = 0

    for entry in tqdm(entries, desc="G2P"):
        try:
            phonemes = g2p(entry["text"])
            entry["phonemes"] = phonemes
            all_ipa_chars.update(phonemes)
        except Exception as e:
            g2p_errors += 1
            entry["phonemes"] = ""

    if g2p_errors:
        print(f"G2P errors: {g2p_errors}")

    # ── Phoneme Audit ───────────────────────────────────────────────────
    sys.path.insert(0, str(TRAINING_DIR))
    from kokoro_symbols import dicts as kokoro_vocab

    covered = set()
    unmapped = set()
    for ch in sorted(all_ipa_chars):
        if ch in kokoro_vocab:
            covered.add(ch)
        else:
            unmapped.add(ch)

    print(f"\n{'=' * 60}")
    print(f"PHONEME AUDIT")
    print(f"{'=' * 60}")
    print(f"Total unique IPA symbols from espeak-ng Turkish: {len(all_ipa_chars)}")
    print(f"Covered by Kokoro vocab: {len(covered)}")
    print(f"UNMAPPED (need fixups):  {len(unmapped)}")

    if unmapped:
        print(f"\nUnmapped symbols:")
        for ch in sorted(unmapped):
            count = sum(1 for e in entries if ch in e.get("phonemes", ""))
            print(f"  {repr(ch):8s} U+{ord(ch):04X}  appears in {count:,} entries")
        print(f"\nAction required: add mappings to PHONEME_FIXUPS in this script,")
        print(f"then re-run with --audit-only to verify.")
    else:
        print(f"\nAll symbols covered — no fixups needed!")

    if audit_only:
        return

    # ── Filter entries with empty phonemes ──────────────────────────────
    valid = [e for e in entries if e.get("phonemes") and len(e["phonemes"]) >= 5]
    print(f"\nValid entries after G2P: {len(valid):,} (dropped {len(entries) - len(valid):,})")

    # ── Train/Val Split ─────────────────────────────────────────────────
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(valid)
    n_val = max(1, int(len(valid) * VAL_RATIO))
    val_entries = valid[:n_val]
    train_entries = valid[n_val:]

    print(f"Split: {len(train_entries):,} train / {len(val_entries):,} val")

    # ── Write StyleTTS2 lists ───────────────────────────────────────────
    # Format: relative_path|phonemes|speaker
    def write_list(path, entry_list):
        with open(path, "w") as f:
            for e in entry_list:
                rel_path = f"{SPEAKER_NAME}/{e['wav_name']}"
                f.write(f"{rel_path}|{e['phonemes']}|{SPEAKER_NAME}\n")

    write_list(TRAIN_LIST, train_entries)
    write_list(VAL_LIST, val_entries)
    print(f"Wrote {TRAIN_LIST} ({len(train_entries):,} lines)")
    print(f"Wrote {VAL_LIST} ({len(val_entries):,} lines)")

    # ── Stats ───────────────────────────────────────────────────────────
    total_dur = sum(e["duration"] for e in valid)
    stats = {
        "speaker": SPEAKER_NAME,
        "total_clips": len(valid),
        "total_duration_h": round(total_dur / 3600, 2),
        "train_clips": len(train_entries),
        "val_clips": len(val_entries),
        "unmapped_phonemes": [repr(ch) for ch in sorted(unmapped)],
    }
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Dataset ready:")
    print(f"  Speaker  : {SPEAKER_NAME}")
    print(f"  Clips    : {stats['total_clips']:,}")
    print(f"  Duration : {stats['total_duration_h']}h")
    print(f"  Train    : {stats['train_clips']:,}")
    print(f"  Val      : {stats['val_clips']:,}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Turkish Kokoro dataset preparation")
    parser.add_argument("--audit-only", action="store_true",
                        help="Only run phoneme audit on existing data")
    args = parser.parse_args()
    download_and_prepare(audit_only=args.audit_only)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create scripts directory and save file**

```bash
mkdir -p scripts
# (write the file above to scripts/prepare_dataset.py)
```

- [ ] **Step 3: Run the dataset preparation**

```bash
uv run python scripts/prepare_dataset.py
```

Expected output:
- Downloads ~9,643 clips (some filtered by duration/word count)
- Resamples all to 24kHz
- Runs G2P on all transcriptions
- Prints phoneme audit showing covered/unmapped symbols
- Writes `training/train_list.txt` and `training/val_list.txt`

**If there are unmapped phonemes**: add them to `PHONEME_FIXUPS` dict in the script with appropriate mappings (nearest Kokoro symbol), then re-run:

```bash
uv run python scripts/prepare_dataset.py --audit-only
```

- [ ] **Step 4: Spot-check a few entries**

```bash
head -5 training/train_list.txt
```

Expected: lines in format `tm_mazlum/000042.wav|ˈmɛɾhɑbɑ...|tm_mazlum`

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_dataset.py
git commit -m "Add Turkish dataset preparation script"
```

---

### Task 4: Write Training Preparation Script

**Files:**
- Create: `scripts/prepare_training.py`

This script converts Kokoro-82M weights to StyleTTS2 format and runs data integrity verification.

- [ ] **Step 1: Write prepare_training.py**

```python
#!/usr/bin/env python3
"""
Turkish Kokoro: Training Preparation
======================================
Converts Kokoro-82M weights to StyleTTS2 format and verifies
training data integrity.

Usage:
    # Convert Kokoro-82M weights
    uv run python scripts/prepare_training.py convert-weights

    # Verify all training data
    uv run python scripts/prepare_training.py verify
"""

import argparse
import json
import sys
from pathlib import Path

TRAINING_DIR = Path("./training")
DATASET_DIR = Path("./dataset")
WAVS_DIR = DATASET_DIR / "audio"
TRAIN_LIST = TRAINING_DIR / "train_list.txt"
VAL_LIST = TRAINING_DIR / "val_list.txt"


def cmd_convert_weights(force: bool = False):
    """Download and convert Kokoro-82M weights to StyleTTS2 format."""
    import torch
    from huggingface_hub import hf_hub_download
    import shutil

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TRAINING_DIR / "kokoro_base.pth"

    if output_path.exists() and not force:
        print(f"Converted weights already exist: {output_path}")
        print("Use --force to regenerate.")
        return

    print("Downloading Kokoro-82M weights from HuggingFace...")
    model_path = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
    config_path = hf_hub_download("hexgrad/Kokoro-82M", "config.json")

    print(f"Loading weights from {model_path}...")
    kokoro_state = torch.load(model_path, map_location="cpu", weights_only=True)

    net = {}
    total_params = 0
    for component, state_dict in kokoro_state.items():
        cleaned = {}
        for key, tensor in state_dict.items():
            clean_key = key.removeprefix("module.")
            cleaned[clean_key] = tensor
            total_params += tensor.numel()
        net[component] = cleaned
        print(f"  {component}: {len(cleaned)} tensors")

    checkpoint = {"net": net}

    config_out = TRAINING_DIR / "config.json"
    shutil.copy2(config_path, config_out)

    torch.save(checkpoint, output_path)
    print(f"\nSaved: {output_path}")
    print(f"  Format: {{'net': {{bert, bert_encoder, predictor, decoder, text_encoder}}}}")
    print(f"Saved: {config_out}")
    print(f"Total parameters: {total_params / 1e6:.2f}M")


def cmd_verify():
    """Verify training data integrity."""
    print("Verifying training data...\n")
    issues = []

    for f in [TRAIN_LIST, VAL_LIST]:
        if not f.exists():
            issues.append(f"MISSING: {f}")

    if issues:
        for issue in issues:
            print(f"  ERROR: {issue}")
        print("\nRun scripts/prepare_dataset.py first.")
        sys.exit(1)

    n_train = 0
    n_val = 0
    missing_wavs = 0
    empty_phonemes = 0
    speakers = set()

    for list_file, label in [(TRAIN_LIST, "train"), (VAL_LIST, "val")]:
        with open(list_file) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) != 3:
                    issues.append(f"{label}:{line_no} — expected 3 fields, got {len(parts)}")
                    continue

                filename, ipa, speaker = parts
                wav_path = WAVS_DIR / filename

                if not wav_path.exists():
                    missing_wavs += 1
                if not ipa or len(ipa) < 3:
                    empty_phonemes += 1
                speakers.add(speaker)

                if label == "train":
                    n_train += 1
                else:
                    n_val += 1

    print(f"  Train entries : {n_train:,}")
    print(f"  Val entries   : {n_val:,}")
    print(f"  Speakers      : {speakers}")
    print(f"  Missing WAVs  : {missing_wavs}")
    print(f"  Empty phonemes: {empty_phonemes}")

    # Check converted weights
    weights_path = TRAINING_DIR / "kokoro_base.pth"
    if weights_path.exists():
        import torch
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        print(f"  Base weights  : OK ({list(state.keys())})")
    else:
        print(f"  Base weights  : NOT FOUND (run 'convert-weights')")

    # Check phoneme vocab coverage
    config_path = TRAINING_DIR / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        vocab = config["vocab"]

        unknown_chars = set()
        with open(TRAIN_LIST) as f:
            for i, line in enumerate(f):
                if i >= 50:
                    break
                ipa = line.strip().split("|")[1]
                for ch in ipa:
                    if ch not in vocab:
                        unknown_chars.add(ch)

        if unknown_chars:
            print(f"  WARNING: Unknown phoneme chars (will be dropped):")
            for ch in sorted(unknown_chars):
                print(f"    {repr(ch)} (U+{ord(ch):04X})")
        else:
            print(f"  Phoneme vocab : OK (all symbols in Kokoro vocab)")

    if issues:
        print(f"\n  ISSUES FOUND:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print(f"\n  All checks passed!")


def main():
    parser = argparse.ArgumentParser(description="Turkish Kokoro training preparation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_convert = subparsers.add_parser("convert-weights", help="Convert Kokoro-82M weights")
    p_convert.add_argument("--force", action="store_true", help="Regenerate if exists")

    subparsers.add_parser("verify", help="Verify training data integrity")

    args = parser.parse_args()
    if args.command == "convert-weights":
        cmd_convert_weights(force=args.force)
    elif args.command == "verify":
        cmd_verify()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run weight conversion**

```bash
uv run python scripts/prepare_training.py convert-weights
```

Expected: downloads Kokoro-82M (~330MB), converts, saves to `training/kokoro_base.pth`. Prints 5 components: bert, bert_encoder, predictor, text_encoder, decoder. Total ~81.76M parameters.

- [ ] **Step 3: Run verification**

```bash
uv run python scripts/prepare_training.py verify
```

Expected: all checks pass — train/val entries present, WAV files exist, phonemes valid, base weights loaded.

- [ ] **Step 4: Commit**

```bash
git add scripts/prepare_training.py
git commit -m "Add training preparation script (weight conversion + verification)"
```

---

### Task 5: Build StyleTTS2 Training Environment

**Files:**
- Modify: `StyleTTS2/text_utils.py` (symlink or patch to use kokoro_symbols)

This task sets up everything StyleTTS2 needs to run training: monotonic alignment, utility models, and the Kokoro symbol mapping.

- [ ] **Step 1: Symlink kokoro_symbols.py into StyleTTS2**

```bash
ln -sf ../../training/kokoro_symbols.py StyleTTS2/kokoro_symbols.py
```

- [ ] **Step 2: Verify StyleTTS2's text_utils.py uses kokoro_symbols**

Check if the patched StyleTTS2 submodule already imports from kokoro_symbols:

```bash
grep -n "kokoro_symbols" StyleTTS2/text_utils.py
```

If not found, patch it:

```bash
# Check current import
head -20 StyleTTS2/text_utils.py
```

The patched semidark/StyleTTS2 fork should already have `from kokoro_symbols import symbols, dicts` in `text_utils.py`. If not, add it by replacing the symbols definition.

- [ ] **Step 3: Build monotonic alignment**

```bash
cd StyleTTS2
git clone https://github.com/resemble-ai/monotonic_align.git
cd monotonic_align
python setup.py build_ext --inplace
cd ../..
```

Expected: builds `monotonic_align/monotonic_align/core.cpython-*.so`

- [ ] **Step 4: Download utility models**

```bash
cd StyleTTS2

# JDC pitch extractor
mkdir -p Utils/JDC
wget -q -O Utils/JDC/bst.t7 "https://github.com/MohammedRakworworke/JDC/releases/download/v1.0/bst.t7" 2>/dev/null || echo "JDC model needs manual download — check TRAINING_GUIDE"

# ASR model
mkdir -p Utils/ASR
# These should already be included in the patched StyleTTS2 submodule
ls Utils/ASR/config.yml Utils/ASR/epoch_00080.pth 2>/dev/null || echo "ASR model needs download"

# PLBERT
ls Utils/PLBERT/ 2>/dev/null || echo "PLBERT needs download"

cd ..
```

Check the kikiri-tts TRAINING_GUIDE for exact download URLs if any are missing. The semidark/StyleTTS2 fork may include these.

- [ ] **Step 5: Install StyleTTS2 training dependencies in a separate venv**

StyleTTS2 training has its own dependencies (accelerate, tensorboard, etc.) that may conflict with the main project. Install them:

```bash
cd StyleTTS2
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install accelerate transformers
pip install librosa soundfile pyyaml tensorboard
pip install munch phonemizer huggingface_hub
cd ..
```

Or use the project's uv environment if compatible.

- [ ] **Step 6: Commit**

```bash
git add StyleTTS2/kokoro_symbols.py
git commit -m "Set up StyleTTS2 training environment"
```

---

### Task 6: Create Training Config

**Files:**
- Create: `configs/config_turkish_ft.yml`

- [ ] **Step 1: Write the Turkish training config**

Adapted from kikiri-tts's `config_german_ft.yml` with Turkish-specific settings:

```yaml
batch_size: 12
epochs: 15
epochs_1st: 15
epochs_2nd: 15
save_freq: 1
pretrained_model: "../training/kokoro_base.pth"
first_stage_path: "first_stage.pth"
load_only_params: true
second_stage_load_pretrained: false

# ─────────────────────────────────────────────────────────────────────────────
# Kokoro Turkish Fine-Tuning Config
# ─────────────────────────────────────────────────────────────────────────────

log_dir: "logs/kokoro-turkish"

data_params:
  train_data: "../training/train_list.txt"
  val_data: "../training/val_list.txt"
  root_path: "../dataset/audio"
  OOD_data: "../training/OOD_texts.txt"
  min_length: 50
  num_workers: 8

preprocess_params:
  sr: 24000
  spect_params:
    n_fft: 2048
    win_length: 1200
    hop_length: 300
    n_mels: 80
    fmin: 0
    fmax: 8000

model_params:
  dim_in: 64
  n_token: 178
  hidden_dim: 512
  style_dim: 128
  max_dur: 50
  multispeaker: false
  n_mels: 80
  dropout: 0.2
  n_layer: 3
  text_encoder_kernel_size: 5

  decoder:
    type: istftnet
    upsample_rates: [10, 6]
    upsample_kernel_sizes: [20, 12]
    upsample_initial_channel: 512
    resblock_kernel_sizes: [3, 7, 11]
    resblock_dilation_sizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    gen_istft_n_fft: 20
    gen_istft_hop_size: 5

  diffusion:
    embedding_mask_proba: 0.1
    transformer:
      num_layers: 3
      num_heads: 8
      head_features: 64
      multiplier: 2
    dist:
      sigma_data: 0.2
      estimate_sigma_data: true
      mean: -3.0
      std: 1.0

  plbert:
    hidden_size: 768
    num_attention_heads: 12
    intermediate_size: 2048
    max_position_embeddings: 512
    num_hidden_layers: 12
    dropout: 0.1

  slm:
    model: "microsoft/wavlm-base-plus"
    sr: 16000
    hidden: 768
    nlayers: 13
    initial_channel: 64

loss_params:
  lambda_gen: 1.0
  lambda_mel: 5.0
  lambda_dur: 1.0
  lambda_ce: 20.0
  lambda_F0: 1.0
  lambda_norm: 1.0
  lambda_s2s: 1.0
  lambda_mono: 1.0
  lambda_slm: 1.0
  lambda_diff: 0.0
  lambda_sty: 0.0
  TMA_epoch: 0
  diff_epoch: 999
  joint_epoch: 3

optimizer_params:
  lr: 0.0001
  bert_lr: 0.00001
  ft_lr: 0.0001

F0_path: "Utils/JDC/bst.t7"
ASR_config: "Utils/ASR/config.yml"
ASR_path: "Utils/ASR/epoch_00080.pth"
PLBERT_dir: "Utils/PLBERT/"

slmadv_params:
  min_len: 100
  max_len: 500
  batch_percentage: 0.5
  iter: 10
  thresh: 5
  scale: 0.01
  sig: 1.5
```

- [ ] **Step 2: Save and verify YAML is valid**

```bash
mkdir -p configs
# (write the file above to configs/config_turkish_ft.yml)
uv run python3 -c "
import yaml
with open('configs/config_turkish_ft.yml') as f:
    config = yaml.safe_load(f)
assert config['batch_size'] == 12
assert config['model_params']['n_token'] == 178
assert config['data_params']['root_path'] == '../dataset/audio'
assert config['pretrained_model'] == '../training/kokoro_base.pth'
print('Config valid.')
print(f'  batch_size: {config[\"batch_size\"]}')
print(f'  epochs_1st: {config[\"epochs_1st\"]}')
print(f'  epochs_2nd: {config[\"epochs_2nd\"]}')
print(f'  lr: {config[\"optimizer_params\"][\"lr\"]}')
"
```

- [ ] **Step 3: Commit**

```bash
git add configs/config_turkish_ft.yml
git commit -m "Add Turkish training config"
```

---

### Task 7: Copy Voicepack Extraction and Inference Scripts

**Files:**
- Copy + adapt: `scripts/extract_voicepack.py` (from kikiri-tts, no changes needed)
- Create: `scripts/test_inference.py` (Turkish test sentences)

- [ ] **Step 1: Copy extract_voicepack.py from kikiri-tts**

```bash
cp /tmp/kikiri-tts/scripts/extract_voicepack.py scripts/extract_voicepack.py
```

If `/tmp/kikiri-tts` no longer exists:

```bash
git clone --depth 1 https://github.com/semidark/kikiri-tts.git /tmp/kikiri-tts
cp /tmp/kikiri-tts/scripts/extract_voicepack.py scripts/extract_voicepack.py
```

No modifications needed — this script is language-agnostic.

- [ ] **Step 2: Write test_inference.py with Turkish test sentences**

```python
#!/usr/bin/env python3
"""
Turkish Kokoro: Test Inference
===============================
Tests the fine-tuned Kokoro model with a Turkish phonetic test set.

Usage:
    # Convert checkpoint + run inference
    uv run python scripts/test_inference.py \
        --checkpoint StyleTTS2/logs/kokoro-turkish/epoch_1st_00002.pth \
        --voicepack voices/tm_mazlum.pt \
        --output-dir test_output/

    # Use a previously converted model
    uv run python scripts/test_inference.py \
        --model voices/kokoro_turkish_converted.pth \
        --voicepack voices/tm_mazlum.pt
"""

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_kokoro_submodule = _repo_root / "kokoro"
if _kokoro_submodule.exists() and str(_kokoro_submodule) not in sys.path:
    sys.path.insert(0, str(_kokoro_submodule))

# Turkish phonetic test set — covers key pronunciation challenges
TEST_SENTENCES = [
    # 1. Vowel harmony (front/back, rounded/unrounded)
    "Güzel bir öğleden sonra, küçük çocuklar bahçede oynuyordu.",
    # 2. ğ behavior (vowel lengthening)
    "Dağdan soğuk rüzgâr doğru yağmur yağdı.",
    # 3. ş/ç/c sibilants
    "Çarşıda şaşkın bir çocuk ceviz çekirdeği çiğniyordu.",
    # 4. Questions and prosody
    "Bu kadar güzel bir gün neden böyle hüzünlü geçiyor?",
    # 5. Numbers and compounds
    "İstanbul'un yedi yüz elli bin nüfuslu ilçesinde yaşıyoruz.",
    # 6. Formal register
    "Türkiye Cumhuriyeti Anayasası'nın birinci maddesi değiştirilemez.",
    # 7. Conversational
    "Hadi gel, bir çay içelim de biraz konuşalım!",
]


def convert_checkpoint(checkpoint_path: str, output_path: str) -> str:
    """Convert StyleTTS2 checkpoint to Kokoro KModel format."""
    import torch

    print(f"Converting checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net = ckpt["net"]

    def ensure_module_prefix(state_dict):
        return {
            ("module." + k if not k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }

    kokoro_weights = {}
    for key in ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"]:
        if key in net:
            kokoro_weights[key] = ensure_module_prefix(net[key])
            print(f"  {key}: {len(kokoro_weights[key])} keys")
        else:
            print(f"  WARNING: '{key}' not found in checkpoint")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kokoro_weights, str(output))
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output} ({size_mb:.1f} MB)")
    return str(output)


def run_inference(
    model_path: str,
    voicepack_path: str,
    config_path: str,
    output_dir: str,
    device: str = "auto",
):
    """Run inference on the Turkish test set."""
    import numpy as np
    import torch
    import soundfile as sf
    from kokoro import KModel, KPipeline

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model from: {model_path}")
    kmodel = KModel(repo_id="hexgrad/Kokoro-82M", config=config_path, model=model_path)
    kmodel = kmodel.to(device).eval()

    # 't' is the Turkish lang_code prefix for Kokoro voice naming
    pipeline = KPipeline(lang_code="t", repo_id="hexgrad/Kokoro-82M", model=kmodel)

    print(f"Loading voicepack: {voicepack_path}")
    voice = torch.load(voicepack_path, map_location="cpu", weights_only=True)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {len(TEST_SENTENCES)} test sentences...\n")
    for i, text in enumerate(TEST_SENTENCES):
        print(f"[{i + 1}/{len(TEST_SENTENCES)}] {text[:60]}...")
        try:
            generator = pipeline(text, voice=voice, speed=1)
            all_audio = []
            for gs, ps, audio in generator:
                print(f"  phonemes: {ps[:60]}...")
                all_audio.append(audio)

            if all_audio:
                combined = np.concatenate(all_audio)
                wav_path = out / f"test_{i + 1:02d}.wav"
                sf.write(str(wav_path), combined, 24000)
                duration = len(combined) / 24000
                print(f"  saved: {wav_path} ({duration:.1f}s)")
            else:
                print(f"  WARNING: No audio generated")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone! Test audio saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned Turkish Kokoro model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="StyleTTS2 checkpoint (.pth) — auto-converted")
    group.add_argument("--model", help="Already-converted Kokoro weights (.pth)")
    parser.add_argument("--voicepack", required=True, help="Voicepack (.pt)")
    parser.add_argument("--config", default="training/config.json", help="Kokoro config.json")
    parser.add_argument("--output-dir", default="test_output/", help="Output WAV directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    args = parser.parse_args()

    if args.checkpoint:
        model_path = convert_checkpoint(
            args.checkpoint,
            str(Path(args.output_dir) / "kokoro_turkish_converted.pth"),
        )
    else:
        model_path = args.model

    run_inference(
        model_path=model_path,
        voicepack_path=args.voicepack,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add scripts/extract_voicepack.py scripts/test_inference.py
git commit -m "Add voicepack extraction and Turkish inference test scripts"
```

---

### Task 8: Smoke Test Training

**Files:** (no new files — runs existing scripts)

Before committing to a full training run, verify the entire pipeline end-to-end with a 2-step smoke test.

- [ ] **Step 1: Set up accelerate config**

```bash
cd StyleTTS2
accelerate config
```

Use the simplest config:
- Compute environment: LOCAL_MACHINE
- Mixed precision: no (fp32)
- Number of processes: 1

- [ ] **Step 2: Run Stage 1 for 2 steps**

```bash
cd StyleTTS2
accelerate launch train_first.py --config_path ../configs/config_turkish_ft.yml
```

Watch the first 2 training steps. Check:
- Model loads without size mismatch errors (missing keys for diffusion/SLM are expected and OK)
- All losses are finite (not NaN or inf)
- Mel Loss is in 0.8–1.5 range

Ctrl+C after confirming losses are healthy.

- [ ] **Step 3: Check for NaN**

If Mel Loss is NaN: symbol mapping is wrong. Verify:

```bash
uv run python3 -c "
import sys
sys.path.insert(0, 'training')
from kokoro_symbols import symbols, dicts
assert len(symbols) == 178
# Check first train entry
with open('training/train_list.txt') as f:
    line = f.readline().strip()
_, phonemes, _ = line.split('|')
unknown = [ch for ch in phonemes if ch not in dicts]
if unknown:
    print(f'UNKNOWN SYMBOLS: {unknown}')
else:
    print(f'All symbols in vocab. Sample: {phonemes[:50]}...')
"
```

- [ ] **Step 4: If smoke test passes, commit a note**

```bash
git add -A
git commit -m "Smoke test passed — training pipeline verified end-to-end"
```

---

### Task 9: Run Full Training

This task is manual (long-running GPU job) but documented for completeness.

- [ ] **Step 1: Stage 1 training (acoustic model)**

```bash
cd StyleTTS2
accelerate launch train_first.py --config_path ../configs/config_turkish_ft.yml
```

Monitor via TensorBoard:

```bash
tensorboard --logdir StyleTTS2/logs/kokoro-turkish --port 6006
```

Expected: ~15 epochs. With batch_size=12 on RTX 5070 Ti, estimate ~2-4 hours per epoch depending on dataset size.

Watch for:
- Mel Loss: 0.8 → 0.25 over 10+ epochs
- Gen/Disc Loss: stable 2.5–4.2
- Mono Loss: < 0.05

- [ ] **Step 2: Stage 2 training (prosody prediction)**

```bash
cd StyleTTS2
accelerate launch train_second.py --config_path ../configs/config_turkish_ft.yml
```

Verify Stage 2 Mel Loss starts at ~0.43, NOT ~7.5. If it starts high, Stage 1 weights were not loaded correctly — check `second_stage_load_pretrained: false` and `first_stage_path` in config.

- [ ] **Step 3: Extract voicepack**

```bash
uv run python scripts/extract_voicepack.py \
    --model StyleTTS2/logs/kokoro-turkish/epoch_2nd_00014.pth \
    --style-encoder-model StyleTTS2/logs/kokoro-turkish/epoch_1st_00014.pth \
    --audio-dir dataset/audio/tm_mazlum \
    --output voices/tm_mazlum.pt
```

Expected: voicepack tensor of shape `[510, 1, 256]`.

- [ ] **Step 4: Test inference**

```bash
uv run python scripts/test_inference.py \
    --checkpoint StyleTTS2/logs/kokoro-turkish/epoch_2nd_00014.pth \
    --voicepack voices/tm_mazlum.pt \
    --output-dir test_output/
```

Listen to the 7 generated WAV files. Check for:
- Intelligible Turkish words
- Turkish prosody (not English-accented)
- Natural-sounding voice matching Mazlum Kiper

---

### Task 10: Update README and CLAUDE.md

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update README.md with full project documentation**

Include: project description, quick start, requirements, training pipeline overview, results (once available), licensing, and credits.

- [ ] **Step 2: Update CLAUDE.md if any commands or conventions changed**

Review the CLAUDE.md against what was actually implemented and fix any discrepancies.

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "Update project documentation"
```
