# Kokoro Voice Studio — Design Spec

## Goal

A web-based tool for recording your voice and creating a custom Kokoro TTS voicepack. Works with any trained Kokoro model and language. Ships with a curated Turkish sentence script; other languages add their own.

## Architecture

**Separate project**: `~/projects/kokoro-voice-studio/`

**Stack**: FastAPI backend + vanilla HTML/JS frontend (single page). Web Audio API for mic capture. WebSocket for real-time status during recording and pipeline execution.

**Relationship to turkish-kokoro**: This tool is a companion — it consumes a trained Kokoro model directory but doesn't depend on the training code. The turkish-kokoro project produces the model; this tool produces voicepacks from it.

## User Flow

1. **Setup**: User launches the app, configures it to point at a trained Kokoro model directory
2. **Create session**: Picks a name (e.g., "cankut"), selects language (e.g., "tr")
3. **Record**: Reads and records sentences one by one via the List+Detail UI
4. **Process**: Chooses Quick (zero-shot voicepack) or Full (fine-tune + voicepack)
5. **Listen**: App plays a test sentence with the new voice

## Recording Interface

**Layout**: List+Detail — sentence list on the left (scrollable, shows recorded/skipped/pending status), recording area on the right with the current sentence displayed prominently.

**Keyboard shortcuts**:
- **Space**: Toggle record/stop. On stop, auto-saves and advances to next sentence
- **R**: Re-record current sentence (overwrites previous take)
- **S**: Skip sentence
- **Left/Right arrows**: Navigate between sentences
- **P**: Play back current recording

**Audio capture**:
- Web Audio API → MediaRecorder
- Capture at browser-native sample rate (typically 48kHz)
- Send as WAV to backend via REST endpoint
- Backend resamples to 24kHz mono 16-bit WAV on save
- Live waveform visualization during recording

**Progress tracking**:
- Each sentence shows status: pending, recorded, skipped
- Counter in header: "48 recorded | 3 skipped | 49 remaining"
- Session state persisted to disk so user can close browser and resume later

## Sentence Scripts

**Format**: JSON files in `sentences/{lang_code}.json`

```json
{
  "language": "tr",
  "language_name": "Turkish",
  "description": "100 sentences covering Turkish phonemes for voice cloning",
  "sentences": [
    {"id": 1, "text": "Güzel bir öğleden sonra, küçük çocuklar bahçede oynuyordu.", "category": "vowel_harmony"},
    {"id": 2, "text": "Dağdan soğuk rüzgâr doğru yağmur yağdı.", "category": "soft_g"}
  ]
}
```

**Shipped with**: Turkish (`tr.json`) — ~100 hand-curated sentences covering all Turkish phonemes, vowel harmony, sibilants, soft-g, questions, exclamations, numbers, formal/informal register. Coverage verified programmatically against espeak-ng phoneme output.

**Adding new languages**: Contributors create their own `sentences/{lang_code}.json` following the documented schema. The README provides the format and guidelines for phoneme coverage.

## Pipeline Modes

### Quick Mode: Zero-Shot Voicepack

1. Validate recordings: check minimum count (≥20), audio quality (duration, silence ratio)
2. Resample all recordings to 24kHz mono WAV (if not already)
3. Run `extract_voicepack.py` against the trained model using the user's recordings
4. Generate a test sentence with the new voicepack
5. Serve audio back to the browser for playback

**Time**: Under 1 minute. **Quality**: Decent approximation — model hasn't trained on this voice.

### Full Mode: Fine-Tune Stage 2

1. Same validation + resample
2. Run G2P on all sentence texts via espeak-ng (language-aware)
3. Apply phoneme fixups for the target language
4. Write StyleTTS2-format train/val lists
5. Fine-tune Stage 2 from the base model's latest checkpoint
6. Stream training progress to browser via WebSocket (epoch, loss, ETA)
7. Extract voicepack from the fine-tuned checkpoint
8. Generate and serve test sample

**Time**: 3-5 hours depending on GPU and recording count. **Quality**: Best possible.

## Configuration

The tool needs to know where the trained Kokoro model lives. Configuration via a YAML file or CLI arguments:

```yaml
# config.yaml
model_dir: /home/orbital/projects/turkish_kokoro
checkpoint: auto  # "auto" picks latest, or specify e.g. "epoch_2nd_00014.pth"
language: tr
```

**What it reads from model_dir**:
- `StyleTTS2/logs/kokoro-turkish/` — checkpoints
- `training/config.json` — Kokoro model config
- `training/kokoro_symbols.py` — phoneme vocab

## Backend API

### REST Endpoints

- `GET /` — Serve the frontend
- `GET /api/session` — Get current session state (sentences, recording status)
- `POST /api/session` — Create new session (name, language)
- `POST /api/recordings/{sentence_id}` — Upload WAV recording for a sentence
- `DELETE /api/recordings/{sentence_id}` — Delete recording (for re-record)
- `GET /api/recordings/{sentence_id}` — Play back a recording
- `POST /api/process/quick` — Run zero-shot voicepack extraction
- `POST /api/process/full` — Run full fine-tune pipeline
- `GET /api/process/status` — Poll pipeline status (or use WebSocket)
- `GET /api/process/sample` — Get generated test audio

### WebSocket

- `/ws/status` — Real-time pipeline progress: `{"epoch": 3, "step": 150, "loss": 0.28, "eta_minutes": 45}`

## Data Layout

```
kokoro-voice-studio/
├── backend/
│   ├── app.py                  # FastAPI application
│   ├── recording.py            # Audio recording session management
│   ├── pipeline.py             # Voicepack extraction + fine-tuning orchestration
│   └── audio.py                # Audio validation + resampling
├── frontend/
│   ├── index.html              # Single-page app
│   ├── recorder.js             # Web Audio API mic capture + keyboard shortcuts
│   └── style.css               # Styles
├── sentences/
│   └── tr.json                 # Turkish sentence script
├── sessions/                   # User recording sessions (gitignored)
│   └── cankut_2026-05-28/
│       ├── session.json        # Session metadata + progress
│       └── recordings/
│           ├── 001.wav
│           ├── 002.wav
│           └── ...
├── config.yaml
├── pyproject.toml
├── README.md
└── CLAUDE.md
```

## Technology

- **Python 3.12+**, managed by uv
- **FastAPI** + uvicorn
- **Web Audio API** for browser mic capture
- **torchaudio** for resampling
- **misaki** / espeak-ng for G2P
- No frontend framework — vanilla HTML/JS/CSS

## Scope Boundaries

**In scope**:
- Recording UI with keyboard shortcuts and session persistence
- Zero-shot voicepack extraction
- Fine-tune Stage 2 pipeline with progress streaming
- Turkish sentence script
- Documentation for adding new languages

**Out of scope**:
- Training a base model from scratch (that's turkish-kokoro's job)
- Multiple simultaneous sessions
- User authentication
- Audio editing/trimming in the UI
- Noise reduction / audio enhancement
