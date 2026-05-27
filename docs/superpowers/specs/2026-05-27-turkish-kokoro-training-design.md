# Turkish Kokoro TTS Training — Design Spec

## Goal

Train Kokoro-82M to produce natural-sounding Turkish speech, starting from a single speaker, for use as the Turkish voice in Pryzm's TTS sidecar. Publish trained weights on HuggingFace.

## Approach

Fork the [kikiri-tts](https://github.com/semidark/kikiri-tts) pipeline (the only successful community Kokoro fine-tune, done for German). Initialize from pretrained English Kokoro-82M weights and train with enough epochs/LR to overwrite English prosody patterns. Single speaker first; expand to multi-speaker if quality is insufficient.

### Why Not Train From Random Init?

Kokoro-82M has 82M parameters. With ~11k clips from a single speaker, random initialization is unlikely to converge. kikiri-tts used 20k clips across 51 speakers and still needed the English checkpoint. The pretrained weights provide a massive head start on decoder quality, mel reconstruction, and general speech structure. Training aggressively (higher LR, more epochs) overwrites the English prosody without losing the structural benefits.

## Dataset

**Primary**: `afkfatih/turkish-tts-combined-raw` on HuggingFace (CC-BY-SA-3.0)

Three named single-speaker subsets:

| Speaker | Clips | Content | Gender |
|---|---|---|---|
| Ahmet Deniz | 11,289 | Religious lectures, formal tone | Male |
| Mazlum Kiper | 9,643 | Literary narration, conversational | Male |
| Nisan Kumru | 8,042 | Audiobook, expressive | Female |

**Starting speaker**: Mazlum Kiper (9,643 clips, literary narration). Nisan Kumru was rejected due to background music throughout the audiobook recordings. Ahmet Deniz was passed over due to narrow domain (religious lectures).

**Audio format**: Source is 48kHz. Resample to 24kHz mono 16-bit WAV during prep using high-quality anti-aliasing (`torchaudio.functional.resample`). Kokoro-82M is architecturally locked to 24kHz (decoder upsampling ratios and hop length are hardcoded).

**Fallback datasets** (if primary speaker quality is insufficient):
- Full afkfatih combined (~81.5k clips, multi-speaker)
- ISSAI Turkish Speech Corpus (218h, CC-BY-4.0)
- Mozilla Common Voice Turkish (134h, CC-0)

## Phoneme Pipeline

### G2P

Use espeak-ng with Turkish language code (`tr`). Turkish is phonetically regular — spelling maps predictably to pronunciation — making G2P cleaner than English or German.

```python
from misaki import espeak

g2p = espeak.EspeakG2P(language='tr')
phonemes, _ = g2p(text)
for old, new in PHONEME_FIXUPS.items():
    phonemes = phonemes.replace(old, new)
```

### Turkish Phoneme Coverage in Kokoro Vocab

Core Turkish phonemes that map to Kokoro's 178-token set:

| Turkish Sound | IPA | Kokoro ID | Status |
|---|---|---|---|
| ş (voiceless postalveolar fricative) | ʃ | 131 | Covered |
| c (voiced postalveolar affricate) | ʤ | 82 | Covered |
| ç (voiceless postalveolar affricate) | ʧ | 133 | Covered |
| ğ (soft g / vowel lengthener) | ː or ∅ | 158 | Handled by espeak |
| ı (close back unrounded) | ɯ | 110 | Covered |
| ö (front rounded mid) | ø | 116 | Covered |
| ü (close front rounded) | y | 67 | Covered |
| r (alveolar tap) | ɾ | 125 | Covered |
| Primary stress | ˈ | 156 | Covered |
| Vowel length | ː | 158 | Covered |

### Phoneme Discovery Pass

Before training, run G2P on all transcriptions, collect the complete set of unique IPA symbols produced, and diff against Kokoro's vocab. Any unmapped symbols get added to `PHONEME_FIXUPS` (map to nearest Kokoro equivalent, same approach as kikiri-tts's `ʏ → y` for German).

### G2P Fallback

misaki's `EspeakG2P` wraps espeak-ng and passes the language code through. Turkish (`tr`) is a fully supported espeak-ng language. If misaki has German-specific post-processing that doesn't generalize, fall back to calling espeak-ng directly via the `phonemizer` Python package.

## Training Pipeline

### Prerequisites

- System: `espeak-ng`, `libsndfile1`, NVIDIA GPU (10GB+ VRAM)
- Python: uv-managed environment (pyproject.toml)
- Submodules: `StyleTTS2/` (semidark/StyleTTS2 patched fork), `kokoro/` (semidark/kokoro fork)

### Stage 0: Dataset Preparation

`scripts/prepare_dataset.py` — simpler than kikiri-tts's version because afkfatih already has transcriptions and identified speakers.

1. **Download**: Stream `afkfatih/turkish-tts-combined-raw` from HuggingFace, filter to chosen speaker
2. **Validate**: Check audio duration (2–30s), text length (≥3 words)
3. **Convert**: Resample to 24kHz mono WAV
4. **G2P**: Run espeak-ng Turkish on each transcription, apply phoneme fixups
5. **Phoneme audit**: Report any IPA symbols not in Kokoro vocab
6. **Write**: StyleTTS2-format lists (`path|phonemes|speaker`), train/val split (95/5)

### Stage 1: Training Preparation

`scripts/prepare_training.py` — reused from kikiri-tts with Turkish adaptations.

1. **Convert weights**: Download Kokoro-82M from HuggingFace, strip `module.` prefix, save as `training/kokoro_base.pth`
2. **Patch StyleTTS2**: Ensure `text_utils.py` imports from `kokoro_symbols.py` (178-token Kokoro mapping)
3. **Build monotonic align**: Compile Cython extension in `StyleTTS2/monotonic_align/`
4. **Download utility models**: JDC pitch extractor, ASR alignment model, PLBERT
5. **Verify**: Run integrity checks on all training data

### Stage 2: Stage 1 Training (Acoustic Model)

Run from `StyleTTS2/`:

```bash
accelerate launch train_first.py --config_path ../configs/config_turkish_ft.yml
```

**Config highlights** (adapted from kikiri-tts German config):
- `batch_size: 12` (fits 16GB RTX 5070 Ti)
- `epochs_1st: 15` (more than German's 10 to overwrite English prosody)
- `multispeaker: false`
- `n_token: 178` (Kokoro vocab)
- `load_only_params: true` (strict=False for missing diffusion/SLM keys)
- `lr: 0.0001`, `bert_lr: 0.00001`

**Healthy loss indicators**:
- Mel Loss: starts 0.8–1.5, drops to 0.25–0.35
- Gen/Disc Loss: stable 2.5–4.2
- Mono Loss: < 0.05
- Any NaN in Mel Loss = symbol mapping problem

### Stage 3: Stage 2 Training (Prosody Prediction)

```bash
accelerate launch train_second.py --config_path ../configs/config_turkish_ft.yml
```

**Key setting**: `second_stage_load_pretrained: false` — loads from Stage 1 checkpoint, not from pretrained Kokoro weights.

**Healthy indicators**:
- Mel Loss starts ~0.43 (proving Stage 1 weights loaded correctly)
- If Mel Loss starts ~7.5–8.0, Stage 1 weights were NOT loaded

**Config additions for Stage 2**:
- `joint_epoch: 3` (start SLM adversarial training)
- `lambda_slm: 1.0` (enable WavLM loss)
- `epochs_2nd: 15`

### Stage 4: Voicepack Extraction

`scripts/extract_voicepack.py` — reused from kikiri-tts without changes.

Averages style vectors (acoustic + prosodic) from ~200 audio samples to produce a voicepack tensor of shape `[510, 1, 256]`.

Recommended: use Stage 1 checkpoint for `style_encoder`, Stage 2 for `predictor_encoder` (Stage 2 can degrade the style encoder).

### Stage 5: Inference Testing

`scripts/test_inference.py` — adapted with Turkish test sentences.

**Turkish phonetic test set** (covers key pronunciation challenges):
1. Vowel harmony (front/back, rounded/unrounded): "Güzel bir öğleden sonra, küçük çocuklar bahçede oynuyordu."
2. ğ behavior (vowel lengthening): "Dağdan soğuk rüzgâr doğru yağmur yağdı."
3. ş/ç/c sibilants: "Çarşıda şaşkın bir çocuk ceviz çekirdeği çiğniyordu."
4. Questions and prosody: "Bu kadar güzel bir gün neden böyle hüzünlü geçiyor?"
5. Numbers and compounds: "İstanbul'un yedi yüz elli bin nüfuslu ilçesinde yaşıyoruz."
6. Formal register: "Türkiye Cumhuriyeti Anayasası'nın birinci maddesi değiştirilemez."
7. Conversational: "Hadi gel, bir çay içelim de biraz konuşalım!"

## OOD Texts

~20 diverse Turkish sentences not present in the training data, converted to IPA phonemes for training regularization. Topics: news, science, daily conversation, literature, geography — ensuring the model generalizes beyond the speaker's domain.

## Repo Structure

```
turkish-kokoro/
├── kokoro/                          # git submodule (semidark/kokoro fork)
├── StyleTTS2/                       # git submodule (semidark/StyleTTS2 patched fork)
├── scripts/
│   ├── prepare_dataset.py           # Download + format afkfatih dataset
│   ├── prepare_training.py          # Train/val split, weight conversion, verify
│   ├── extract_voicepack.py         # Voicepack extraction (from kikiri-tts)
│   └── test_inference.py            # Turkish test sentences + inference
├── configs/
│   └── config_turkish_ft.yml        # Training hyperparameters
├── training/
│   ├── kokoro_symbols.py            # 178-token Kokoro vocab (from kikiri-tts)
│   ├── config.json                  # Kokoro model config
│   ├── OOD_texts.txt                # Turkish OOD phoneme sequences
│   ├── train_list.txt               # Generated
│   └── val_list.txt                 # Generated
├── dataset/
│   └── audio/                       # Downloaded + resampled WAVs
├── voices/                          # Extracted voicepacks
├── samples/                         # Speaker audition samples
├── docs/
│   ├── TRAINING_GUIDE.md            # Turkish-specific training guide
│   └── superpowers/specs/           # This spec
├── pyproject.toml
├── CLAUDE.md
├── README.md
└── .gitmodules
```

## Evaluation

1. **Loss curves**: Monitor via TensorBoard during training
2. **Listening tests**: Extract voicepacks and generate test sentences after each epoch
3. **Baseline comparison**: Compare against English Kokoro on Turkish text (should be clearly better)
4. **Intelligibility**: Can a Turkish speaker understand every word?
5. **Naturalness**: Does prosody sound Turkish, not English-accented?

## Integration with Pryzm

Once the model meets the quality bar:
1. Export trained checkpoint to Kokoro-compatible format (5 components: bert, bert_encoder, predictor, text_encoder, decoder)
2. The Kokoro FastAPI sidecar in Pryzm already supports language parameters — swap in the Turkish model weights
3. `synthesize.py` in Pryzm backend already passes `language` param to Kokoro
4. Voicepack goes alongside English voices in the Kokoro container

## Custom Voice Addition (Future)

Two paths for adding a new voice (e.g., the user's own voice) after the base Turkish model is trained:

### Path A: Zero-Shot Voicepack (Minutes, No Training)

1. Record 5–10 minutes of diverse Turkish speech
2. Convert to 24kHz mono WAV
3. Run `extract_voicepack.py` against the trained Turkish model — this uses the model's style encoder to extract a voicepack from the new audio
4. Test with `test_inference.py`

Quality: decent but limited — the model hasn't heard this voice during training, so it approximates.

### Path B: Fine-Tune Stage 2 (Hours, Best Quality)

1. Record 30–60 minutes of Turkish speech (provided script for phonetic coverage)
2. Run dataset prep + G2P
3. Fine-tune Stage 2 only from the trained Turkish base model (Stage 1 stays frozen)
4. Extract voicepack from the fine-tuned checkpoint

Quality: best possible — the model learns the voice's specific characteristics.

Both paths are straightforward because the training pipeline and tools are already built. Path A is try-it-now, Path B is invest-for-quality.

### Privacy Note

Publishing a voicepack trained on your own voice effectively makes your voice clonable. The base Turkish model weights can be published openly, but personal voicepacks should stay private unless you're comfortable with that.

## Licensing

- **Training code**: Apache-2.0 (matching kikiri-tts)
- **Model weights**: CC-BY-SA-3.0 (inherited from afkfatih dataset)
- **Personal voicepacks**: Private by default

## Known Risks

1. **English prosody bleed-through**: Mitigated by training aggressively (more epochs, reasonable LR). If persistent, escalate to multi-speaker training with more data.
2. **Phoneme mapping gaps**: Mitigated by discovery pass before training. Turkish IPA maps well to Kokoro's vocab.
3. **Pioneer risk**: No one has done Turkish Kokoro before. Expect iteration.
4. **Speaker domain bias**: Mazlum Kiper's literary narration covers conversational, dramatic, and descriptive registers, which should generalize reasonably well. If the model struggles with specific styles (e.g., casual chat), augmenting with clips from other speakers may help.
