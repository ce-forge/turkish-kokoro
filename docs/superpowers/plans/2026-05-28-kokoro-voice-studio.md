# Kokoro Voice Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a web-based voice recording tool that lets users create custom Kokoro TTS voicepacks — either zero-shot (instant) or via Stage 2 fine-tuning (best quality).

**Architecture:** FastAPI backend serving a single-page vanilla HTML/JS frontend. Web Audio API captures mic input, backend handles audio resampling, session persistence, and orchestrates the Kokoro pipeline (voicepack extraction or fine-tuning). The tool is model-agnostic — it points at any trained Kokoro model directory.

**Tech Stack:** Python 3.12 (uv), FastAPI, uvicorn, torchaudio, misaki/espeak-ng, vanilla HTML/JS/CSS, Web Audio API, WebSocket

---

### Task 1: Project Scaffold

**Files:**
- Create: `~/projects/kokoro-voice-studio/pyproject.toml`
- Create: `~/projects/kokoro-voice-studio/README.md`
- Create: `~/projects/kokoro-voice-studio/CLAUDE.md`
- Create: `~/projects/kokoro-voice-studio/.gitignore`
- Create: `~/projects/kokoro-voice-studio/config.yaml`

- [ ] **Step 1: Create project directory and initialize git**

```bash
mkdir -p ~/projects/kokoro-voice-studio
cd ~/projects/kokoro-voice-studio
git init
```

- [ ] **Step 2: Write pyproject.toml**

```toml
[project]
name = "kokoro-voice-studio"
version = "0.1.0"
description = "Record your voice, create a custom Kokoro TTS voicepack"
readme = "README.md"
license = { text = "Apache-2.0" }
requires-python = ">=3.10, <3.14"
dependencies = [
    "fastapi",
    "uvicorn[standard]",
    "python-multipart",
    "pyyaml",
    "torch",
    "torchaudio",
    "soundfile",
    "numpy",
    "misaki[en]>=0.9.4",
    "huggingface_hub",
    "websockets",
]

[tool.uv]
find-links = ["https://download.pytorch.org/whl/cu128"]

[tool.uv.sources]
misaki = { git = "https://github.com/semidark/misaki.git", branch = "main" }
```

- [ ] **Step 3: Write .gitignore**

```
.venv/
sessions/
__pycache__/
*.pyc
.superpowers/
```

- [ ] **Step 4: Write config.yaml**

```yaml
# Path to a trained Kokoro model project directory
# Must contain: StyleTTS2/logs/*, training/config.json, training/kokoro_symbols.py
model_dir: /home/orbital/projects/turkish_kokoro
language: tr
host: 127.0.0.1
port: 8888
```

- [ ] **Step 5: Write minimal README.md**

```markdown
# Kokoro Voice Studio

Record your voice and create a custom Kokoro TTS voicepack.

## Quick Start

\`\`\`bash
uv sync
uv run python -m backend.app
# Open http://localhost:8888
\`\`\`

## Configuration

Edit `config.yaml` to point at your trained Kokoro model directory.

## Adding Languages

Create `sentences/{lang_code}.json` following the format in `sentences/tr.json`.
\`\`\`
```

- [ ] **Step 6: Write CLAUDE.md**

```markdown
# Kokoro Voice Studio

Web-based voice recording + voicepack creation tool for Kokoro TTS.

## Architecture

- FastAPI backend (`backend/`) serves the frontend and handles audio + pipeline
- Vanilla HTML/JS frontend (`frontend/`) — single page, no framework
- Sentence scripts (`sentences/`) — JSON files per language

## Commands

\`\`\`bash
uv sync
uv run python -m backend.app          # Start server
uv run pytest tests/ -v                # Run tests
\`\`\`

## Key Conventions

- Audio is always stored as 24kHz mono 16-bit WAV
- Sessions persist to `sessions/{name}_{date}/`
- Config in `config.yaml` points to a trained Kokoro model directory
```

- [ ] **Step 7: Sync environment**

```bash
uv sync
```

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "Initial project scaffold"
```

---

### Task 2: Audio Utilities

**Files:**
- Create: `backend/__init__.py`
- Create: `backend/audio.py`
- Create: `tests/__init__.py`
- Create: `tests/test_audio.py`

- [ ] **Step 1: Create directories**

```bash
mkdir -p backend tests
touch backend/__init__.py tests/__init__.py
```

- [ ] **Step 2: Write test for resample function**

```python
# tests/test_audio.py
import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path
from backend.audio import resample_to_24k, validate_recording


def test_resample_48k_to_24k():
    """48kHz input should be resampled to 24kHz."""
    # Generate 1 second of 48kHz sine wave
    sr_in = 48000
    t = np.linspace(0, 1, sr_in, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr_in)
        input_path = Path(f.name)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        output_path = Path(f.name)

    resample_to_24k(input_path, output_path)

    data, sr = sf.read(str(output_path))
    assert sr == 24000
    assert len(data) == 24000  # 1 second at 24kHz
    assert data.ndim == 1  # mono

    input_path.unlink()
    output_path.unlink()


def test_resample_already_24k():
    """24kHz input should be copied without distortion."""
    sr_in = 24000
    t = np.linspace(0, 1, sr_in, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr_in)
        input_path = Path(f.name)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        output_path = Path(f.name)

    resample_to_24k(input_path, output_path)

    data, sr = sf.read(str(output_path))
    assert sr == 24000
    assert len(data) == 24000

    input_path.unlink()
    output_path.unlink()


def test_validate_recording_good():
    """A 3-second recording should pass validation."""
    sr = 24000
    t = np.linspace(0, 3, sr * 3, dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 200 * t)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr)
        path = Path(f.name)

    result = validate_recording(path)
    assert result["valid"] is True
    path.unlink()


def test_validate_recording_too_short():
    """A 0.5-second recording should fail."""
    sr = 24000
    audio = np.zeros(int(sr * 0.5), dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr)
        path = Path(f.name)

    result = validate_recording(path)
    assert result["valid"] is False
    assert "short" in result["reason"]
    path.unlink()


def test_validate_recording_silent():
    """A silent recording should fail."""
    sr = 24000
    audio = np.zeros(sr * 3, dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr)
        path = Path(f.name)

    result = validate_recording(path)
    assert result["valid"] is False
    assert "silent" in result["reason"]
    path.unlink()
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_audio.py -v
```

Expected: FAIL — `backend.audio` doesn't exist yet.

- [ ] **Step 4: Write backend/audio.py**

```python
"""Audio utilities: resampling and validation."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

TARGET_SR = 24000
MIN_DURATION_S = 1.0
MIN_RMS = 0.005


def resample_to_24k(input_path: Path, output_path: Path) -> None:
    """Resample audio to 24kHz mono 16-bit WAV."""
    data, sr = sf.read(str(input_path), dtype="float32")

    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != TARGET_SR:
        waveform = torch.from_numpy(data).unsqueeze(0)
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        data = waveform.squeeze(0).numpy()

    sf.write(str(output_path), data, TARGET_SR, subtype="PCM_16")


def validate_recording(path: Path) -> dict:
    """Check a recording for minimum duration and audio level."""
    data, sr = sf.read(str(path), dtype="float32")
    duration = len(data) / sr

    if duration < MIN_DURATION_S:
        return {"valid": False, "reason": "too short", "duration": duration}

    rms = float(np.sqrt(np.mean(data**2)))
    if rms < MIN_RMS:
        return {"valid": False, "reason": "silent", "rms": rms}

    return {"valid": True, "duration": duration, "rms": rms}
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_audio.py -v
```

Expected: all 5 pass.

- [ ] **Step 6: Commit**

```bash
git add backend/ tests/
git commit -m "Add audio utilities (resample + validation)"
```

---

### Task 3: Session Management

**Files:**
- Create: `backend/session.py`
- Create: `tests/test_session.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_session.py
import json
import tempfile
from pathlib import Path
from backend.session import Session, load_sentences


def test_load_sentences():
    """Load a sentence script from JSON."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump({
            "language": "tr",
            "language_name": "Turkish",
            "description": "Test",
            "sentences": [
                {"id": 1, "text": "Merhaba dünya.", "category": "greeting"},
                {"id": 2, "text": "Nasılsın?", "category": "question"},
            ]
        }, f)
        path = Path(f.name)

    sentences = load_sentences(path)
    assert len(sentences) == 2
    assert sentences[0]["text"] == "Merhaba dünya."
    path.unlink()


def test_create_session():
    """Create a new session with correct initial state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        session = Session.create(
            name="testuser",
            language="tr",
            sentences=[
                {"id": 1, "text": "Merhaba.", "category": "greeting"},
                {"id": 2, "text": "Nasılsın?", "category": "question"},
            ],
            base_dir=Path(tmpdir),
        )

        assert session.name == "testuser"
        assert session.language == "tr"
        assert len(session.sentences) == 2
        assert session.recordings_dir.exists()
        assert session.status(1) == "pending"
        assert session.status(2) == "pending"
        assert session.stats() == {"recorded": 0, "skipped": 0, "pending": 2}


def test_mark_recorded():
    """Marking a sentence as recorded updates status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        session = Session.create(
            name="testuser",
            language="tr",
            sentences=[{"id": 1, "text": "Test.", "category": "test"}],
            base_dir=Path(tmpdir),
        )
        session.mark_recorded(1)
        assert session.status(1) == "recorded"
        assert session.stats() == {"recorded": 1, "skipped": 0, "pending": 0}


def test_mark_skipped():
    with tempfile.TemporaryDirectory() as tmpdir:
        session = Session.create(
            name="testuser",
            language="tr",
            sentences=[{"id": 1, "text": "Test.", "category": "test"}],
            base_dir=Path(tmpdir),
        )
        session.mark_skipped(1)
        assert session.status(1) == "skipped"


def test_session_persistence():
    """Session state survives save/load cycle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        session = Session.create(
            name="testuser",
            language="tr",
            sentences=[
                {"id": 1, "text": "A.", "category": "test"},
                {"id": 2, "text": "B.", "category": "test"},
            ],
            base_dir=Path(tmpdir),
        )
        session.mark_recorded(1)
        session.save()

        loaded = Session.load(session.session_dir)
        assert loaded.status(1) == "recorded"
        assert loaded.status(2) == "pending"
        assert loaded.stats() == {"recorded": 1, "skipped": 0, "pending": 1}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_session.py -v
```

- [ ] **Step 3: Write backend/session.py**

```python
"""Recording session management with disk persistence."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from dataclasses import dataclass, field


def load_sentences(path: Path) -> list[dict]:
    """Load sentence script from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data["sentences"]


@dataclass
class Session:
    name: str
    language: str
    sentences: list[dict]
    session_dir: Path
    recordings_dir: Path
    _statuses: dict[int, str] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        language: str,
        sentences: list[dict],
        base_dir: Path,
    ) -> Session:
        dir_name = f"{name}_{date.today().isoformat()}"
        session_dir = base_dir / dir_name
        recordings_dir = session_dir / "recordings"
        recordings_dir.mkdir(parents=True, exist_ok=True)

        statuses = {s["id"]: "pending" for s in sentences}
        session = cls(
            name=name,
            language=language,
            sentences=sentences,
            session_dir=session_dir,
            recordings_dir=recordings_dir,
            _statuses=statuses,
        )
        session.save()
        return session

    @classmethod
    def load(cls, session_dir: Path) -> Session:
        with open(session_dir / "session.json") as f:
            data = json.load(f)
        return cls(
            name=data["name"],
            language=data["language"],
            sentences=data["sentences"],
            session_dir=session_dir,
            recordings_dir=session_dir / "recordings",
            _statuses=data["statuses"],
        )

    def save(self) -> None:
        data = {
            "name": self.name,
            "language": self.language,
            "sentences": self.sentences,
            "statuses": self._statuses,
        }
        with open(self.session_dir / "session.json", "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def status(self, sentence_id: int) -> str:
        return self._statuses.get(sentence_id, "pending")

    def mark_recorded(self, sentence_id: int) -> None:
        self._statuses[sentence_id] = "recorded"
        self.save()

    def mark_skipped(self, sentence_id: int) -> None:
        self._statuses[sentence_id] = "skipped"
        self.save()

    def recording_path(self, sentence_id: int) -> Path:
        return self.recordings_dir / f"{sentence_id:03d}.wav"

    def stats(self) -> dict:
        statuses = list(self._statuses.values())
        return {
            "recorded": statuses.count("recorded"),
            "skipped": statuses.count("skipped"),
            "pending": statuses.count("pending"),
        }
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_session.py -v
```

Expected: all 5 pass.

- [ ] **Step 5: Commit**

```bash
git add backend/session.py tests/test_session.py
git commit -m "Add session management with persistence"
```

---

### Task 4: FastAPI Backend

**Files:**
- Create: `backend/app.py`
- Create: `backend/config.py`

- [ ] **Step 1: Write backend/config.py**

```python
"""Application configuration loaded from config.yaml."""

from pathlib import Path

import yaml

CONFIG_PATH = Path("config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_model_dir(config: dict) -> Path:
    return Path(config["model_dir"])


def get_sentences_path(config: dict) -> Path:
    lang = config["language"]
    return Path("sentences") / f"{lang}.json"
```

- [ ] **Step 2: Write backend/app.py**

```python
"""FastAPI application — serves frontend, handles recordings, runs pipeline."""

from __future__ import annotations

import io
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.audio import resample_to_24k, validate_recording
from backend.config import load_config, get_sentences_path
from backend.session import Session, load_sentences

app = FastAPI(title="Kokoro Voice Studio")
config = load_config()

SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

_session: Session | None = None


class CreateSessionRequest(BaseModel):
    name: str
    language: str | None = None


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse("frontend/index.html")


app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/api/sentences")
def get_sentences():
    path = get_sentences_path(config)
    if not path.exists():
        raise HTTPException(404, f"No sentence script for language: {config['language']}")
    sentences = load_sentences(path)
    return {"language": config["language"], "sentences": sentences}


@app.post("/api/session")
def create_session(req: CreateSessionRequest):
    global _session
    lang = req.language or config["language"]
    lang_path = Path("sentences") / f"{lang}.json"
    if not lang_path.exists():
        raise HTTPException(404, f"No sentence script for language: {lang}")
    sentences = load_sentences(lang_path)
    _session = Session.create(
        name=req.name,
        language=lang,
        sentences=sentences,
        base_dir=SESSIONS_DIR,
    )
    return {"session_dir": str(_session.session_dir), "stats": _session.stats()}


@app.get("/api/session")
def get_session():
    if _session is None:
        raise HTTPException(404, "No active session")
    return {
        "name": _session.name,
        "language": _session.language,
        "sentences": _session.sentences,
        "statuses": _session._statuses,
        "stats": _session.stats(),
    }


@app.post("/api/recordings/{sentence_id}")
async def upload_recording(sentence_id: int, file: UploadFile):
    if _session is None:
        raise HTTPException(404, "No active session")

    raw_bytes = await file.read()
    raw_path = _session.recordings_dir / f"{sentence_id:03d}_raw.wav"
    final_path = _session.recording_path(sentence_id)

    with open(raw_path, "wb") as f:
        f.write(raw_bytes)

    resample_to_24k(raw_path, final_path)
    raw_path.unlink()

    validation = validate_recording(final_path)
    if not validation["valid"]:
        final_path.unlink()
        return {"status": "rejected", **validation}

    _session.mark_recorded(sentence_id)
    return {"status": "ok", **validation}


@app.delete("/api/recordings/{sentence_id}")
def delete_recording(sentence_id: int):
    if _session is None:
        raise HTTPException(404, "No active session")
    path = _session.recording_path(sentence_id)
    if path.exists():
        path.unlink()
    _session._statuses[sentence_id] = "pending"
    _session.save()
    return {"status": "ok"}


@app.get("/api/recordings/{sentence_id}")
def get_recording(sentence_id: int):
    if _session is None:
        raise HTTPException(404, "No active session")
    path = _session.recording_path(sentence_id)
    if not path.exists():
        raise HTTPException(404, "No recording")
    return FileResponse(str(path), media_type="audio/wav")


@app.post("/api/recordings/{sentence_id}/skip")
def skip_sentence(sentence_id: int):
    if _session is None:
        raise HTTPException(404, "No active session")
    _session.mark_skipped(sentence_id)
    return {"status": "ok"}


def main():
    import uvicorn
    uvicorn.run(app, host=config.get("host", "127.0.0.1"), port=config.get("port", 8888))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add `__main__.py` for `python -m backend.app`**

```python
# backend/__main__.py
from backend.app import main
main()
```

- [ ] **Step 4: Verify server starts**

```bash
uv run python -m backend.app &
sleep 2
curl -s http://localhost:8888/api/sentences | head -c 200
kill %1
```

Expected: returns 404 (no sentence file yet) or the sentences JSON if `sentences/tr.json` exists.

- [ ] **Step 5: Commit**

```bash
git add backend/
git commit -m "Add FastAPI backend with recording endpoints"
```

---

### Task 5: Frontend Recording UI

**Files:**
- Create: `frontend/index.html`
- Create: `frontend/style.css`
- Create: `frontend/recorder.js`

- [ ] **Step 1: Write frontend/index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kokoro Voice Studio</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <!-- Setup screen -->
    <div id="setup-screen">
        <h1>Kokoro Voice Studio</h1>
        <p>Record your voice to create a custom TTS voicepack.</p>
        <div class="setup-form">
            <label>Your name:
                <input type="text" id="session-name" placeholder="e.g. cankut" autofocus>
            </label>
            <button id="start-btn" onclick="startSession()">Start Recording Session</button>
        </div>
    </div>

    <!-- Recording screen -->
    <div id="recording-screen" style="display:none;">
        <header>
            <h2 id="session-title"></h2>
            <div id="stats"></div>
        </header>
        <div class="main-layout">
            <aside id="sentence-list"></aside>
            <main>
                <div id="current-sentence"></div>
                <div id="waveform"></div>
                <div class="controls">
                    <button id="play-btn" onclick="playBack()" disabled>▶ Play</button>
                    <button id="record-btn" class="record-btn">⏺ Space to Record</button>
                    <button id="rerecord-btn" onclick="reRecord()" disabled>↺ Re-record</button>
                    <button id="skip-btn" onclick="skipSentence()">Skip →</button>
                </div>
                <div class="shortcuts">
                    Space: Record/Stop &nbsp;|&nbsp; R: Re-record &nbsp;|&nbsp; S: Skip &nbsp;|&nbsp; ←→: Navigate &nbsp;|&nbsp; P: Play
                </div>
            </main>
        </div>
        <!-- Done overlay -->
        <div id="done-screen" style="display:none;">
            <h2>Recording Complete!</h2>
            <div id="done-stats"></div>
            <div class="done-actions">
                <button onclick="processQuick()">⚡ Quick Voicepack (instant)</button>
                <button onclick="processFull()">🎯 Full Fine-tune (best quality)</button>
            </div>
            <div id="process-status" style="display:none;"></div>
            <audio id="sample-audio" controls style="display:none;"></audio>
        </div>
    </div>
    <script src="/static/recorder.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write frontend/style.css**

```css
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: system-ui, -apple-system, sans-serif;
    background: #0d1117;
    color: #e0e0e0;
    height: 100vh;
    overflow: hidden;
}

/* Setup */
#setup-screen {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    gap: 16px;
}
#setup-screen h1 { font-size: 2em; }
.setup-form {
    display: flex;
    flex-direction: column;
    gap: 12px;
    width: 300px;
}
.setup-form input {
    padding: 10px;
    border-radius: 6px;
    border: 1px solid #333;
    background: #161b22;
    color: #e0e0e0;
    font-size: 16px;
}
.setup-form button {
    padding: 12px;
    border-radius: 6px;
    border: none;
    background: #238636;
    color: white;
    font-size: 16px;
    cursor: pointer;
}
.setup-form button:hover { background: #2ea043; }

/* Header */
header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 20px;
    background: #161b22;
    border-bottom: 1px solid #21262d;
}
#stats {
    background: #21262d;
    padding: 6px 14px;
    border-radius: 16px;
    font-size: 13px;
}

/* Main layout */
.main-layout {
    display: flex;
    height: calc(100vh - 56px);
}

/* Sentence list */
#sentence-list {
    width: 280px;
    overflow-y: auto;
    background: #161b22;
    border-right: 1px solid #21262d;
    padding: 8px;
}
.sentence-item {
    padding: 8px 10px;
    border-radius: 6px;
    font-size: 12px;
    margin-bottom: 4px;
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.sentence-item.pending { background: #0d1117; color: #666; }
.sentence-item.recorded { background: #1a2e1a; color: #4ade80; }
.sentence-item.skipped { background: #2e2a1a; color: #f59e0b; }
.sentence-item.active { border-left: 3px solid #58a6ff; background: #1c2333; color: #e0e0e0; }

/* Recording area */
main {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 24px;
    padding: 40px;
}
#current-sentence {
    font-size: 24px;
    line-height: 1.6;
    text-align: center;
    max-width: 700px;
    background: #161b22;
    padding: 32px;
    border-radius: 12px;
}
#waveform {
    width: 100%;
    max-width: 600px;
    height: 60px;
    background: #161b22;
    border-radius: 8px;
}

/* Controls */
.controls {
    display: flex;
    gap: 12px;
    align-items: center;
}
.controls button {
    padding: 10px 20px;
    border-radius: 6px;
    border: none;
    background: #21262d;
    color: #c9d1d9;
    font-size: 14px;
    cursor: pointer;
}
.controls button:hover { background: #30363d; }
.controls button:disabled { opacity: 0.3; cursor: default; }
.record-btn {
    padding: 14px 28px !important;
    border-radius: 24px !important;
    background: #21262d !important;
    font-size: 16px !important;
}
.record-btn.recording {
    background: #da3633 !important;
    color: white !important;
    animation: pulse 1s infinite;
}
@keyframes pulse { 50% { opacity: 0.7; } }

.shortcuts {
    font-size: 12px;
    color: #484f58;
}

/* Done screen */
#done-screen {
    position: fixed;
    inset: 0;
    background: rgba(13,17,23,0.95);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 20px;
}
.done-actions {
    display: flex;
    gap: 16px;
}
.done-actions button {
    padding: 16px 32px;
    border-radius: 8px;
    border: none;
    font-size: 16px;
    cursor: pointer;
    background: #238636;
    color: white;
}
.done-actions button:hover { background: #2ea043; }
```

- [ ] **Step 3: Write frontend/recorder.js**

```javascript
let session = null;
let sentences = [];
let currentIndex = 0;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

async function startSession() {
    const name = document.getElementById('session-name').value.trim();
    if (!name) return;

    const res = await fetch('/api/session', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name})
    });
    session = await res.json();

    const sessionRes = await fetch('/api/session');
    const data = await sessionRes.json();
    sentences = data.sentences;
    session.statuses = data.statuses;

    document.getElementById('setup-screen').style.display = 'none';
    document.getElementById('recording-screen').style.display = 'block';
    document.getElementById('session-title').textContent = `Recording: ${name}`;

    renderList();
    showSentence(0);
    updateStats();
}

function renderList() {
    const list = document.getElementById('sentence-list');
    list.innerHTML = sentences.map((s, i) => {
        const status = session.statuses[s.id] || 'pending';
        const prefix = status === 'recorded' ? '✓' : status === 'skipped' ? '–' : '';
        return `<div class="sentence-item ${status} ${i === currentIndex ? 'active' : ''}"
                     onclick="showSentence(${i})"
                     id="item-${i}">
                    ${prefix} ${s.text}
                </div>`;
    }).join('');
}

function showSentence(index) {
    if (index < 0 || index >= sentences.length) return;
    currentIndex = index;
    const s = sentences[index];
    document.getElementById('current-sentence').textContent = `"${s.text}"`;

    const hasRecording = session.statuses[s.id] === 'recorded';
    document.getElementById('play-btn').disabled = !hasRecording;
    document.getElementById('rerecord-btn').disabled = !hasRecording;

    renderList();
    document.getElementById(`item-${index}`)?.scrollIntoView({block: 'nearest'});
}

function updateStats() {
    const statuses = Object.values(session.statuses);
    const recorded = statuses.filter(s => s === 'recorded').length;
    const skipped = statuses.filter(s => s === 'skipped').length;
    const pending = statuses.filter(s => s === 'pending').length;
    document.getElementById('stats').innerHTML =
        `<span style="color:#4ade80">● ${recorded} recorded</span> | ` +
        `<span style="color:#f59e0b">● ${skipped} skipped</span> | ` +
        `${pending} remaining`;

    if (pending === 0 && recorded > 0) {
        document.getElementById('done-screen').style.display = 'flex';
        document.getElementById('done-stats').textContent =
            `${recorded} recorded, ${skipped} skipped`;
    }
}

async function toggleRecord() {
    if (isRecording) {
        mediaRecorder.stop();
    } else {
        const stream = await navigator.mediaDevices.getUserMedia({audio: true});
        mediaRecorder = new MediaRecorder(stream, {mimeType: 'audio/webm'});
        audioChunks = [];

        mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach(t => t.stop());
            const blob = new Blob(audioChunks, {type: 'audio/webm'});
            await uploadRecording(blob);
        };

        mediaRecorder.start();
        isRecording = true;
        document.getElementById('record-btn').classList.add('recording');
        document.getElementById('record-btn').textContent = '⏹ Recording...';
    }
}

async function uploadRecording(blob) {
    const sid = sentences[currentIndex].id;
    const form = new FormData();
    form.append('file', blob, `${sid}.webm`);

    const res = await fetch(`/api/recordings/${sid}`, {method: 'POST', body: form});
    const result = await res.json();

    isRecording = false;
    document.getElementById('record-btn').classList.remove('recording');
    document.getElementById('record-btn').textContent = '⏺ Space to Record';

    if (result.status === 'ok') {
        session.statuses[sid] = 'recorded';
        updateStats();
        // Auto-advance to next pending sentence
        const nextIndex = sentences.findIndex((s, i) =>
            i > currentIndex && session.statuses[s.id] === 'pending');
        if (nextIndex !== -1) {
            showSentence(nextIndex);
        } else {
            renderList();
        }
    }
}

async function playBack() {
    const sid = sentences[currentIndex].id;
    const audio = new Audio(`/api/recordings/${sid}`);
    audio.play();
}

async function reRecord() {
    const sid = sentences[currentIndex].id;
    await fetch(`/api/recordings/${sid}`, {method: 'DELETE'});
    session.statuses[sid] = 'pending';
    updateStats();
    showSentence(currentIndex);
    toggleRecord();
}

async function skipSentence() {
    const sid = sentences[currentIndex].id;
    await fetch(`/api/recordings/${sid}/skip`, {method: 'POST'});
    session.statuses[sid] = 'skipped';
    updateStats();

    const nextIndex = sentences.findIndex((s, i) =>
        i > currentIndex && session.statuses[s.id] === 'pending');
    if (nextIndex !== -1) showSentence(nextIndex);
}

async function processQuick() {
    document.getElementById('process-status').style.display = 'block';
    document.getElementById('process-status').textContent = 'Extracting voicepack...';
    const res = await fetch('/api/process/quick', {method: 'POST'});
    const result = await res.json();
    document.getElementById('process-status').textContent = result.message || 'Done!';
    if (result.sample_url) {
        const audio = document.getElementById('sample-audio');
        audio.src = result.sample_url;
        audio.style.display = 'block';
    }
}

async function processFull() {
    document.getElementById('process-status').style.display = 'block';
    document.getElementById('process-status').textContent = 'Starting fine-tuning... this will take several hours.';
    const res = await fetch('/api/process/full', {method: 'POST'});
    const result = await res.json();
    document.getElementById('process-status').textContent = result.message || 'Started.';
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;

    switch(e.code) {
        case 'Space':
            e.preventDefault();
            toggleRecord();
            break;
        case 'KeyR':
            if (!isRecording && session?.statuses[sentences[currentIndex]?.id] === 'recorded')
                reRecord();
            break;
        case 'KeyS':
            if (!isRecording) skipSentence();
            break;
        case 'KeyP':
            if (!isRecording) playBack();
            break;
        case 'ArrowLeft':
            showSentence(currentIndex - 1);
            break;
        case 'ArrowRight':
            showSentence(currentIndex + 1);
            break;
    }
});
```

- [ ] **Step 4: Test the UI manually**

```bash
uv run python -m backend.app
```

Open http://localhost:8888 in browser. Verify:
- Setup screen appears
- Entering a name and clicking start shows the recording screen (will fail until sentences/tr.json exists — that's Task 6)

- [ ] **Step 5: Commit**

```bash
git add frontend/
git commit -m "Add recording frontend (List+Detail UI with keyboard shortcuts)"
```

---

### Task 6: Turkish Sentence Script

**Files:**
- Create: `sentences/tr.json`

- [ ] **Step 1: Create sentences directory**

```bash
mkdir -p sentences
```

- [ ] **Step 2: Write sentences/tr.json**

Generate ~100 Turkish sentences covering all phonemes. Use espeak-ng to verify coverage.

The sentences should be curated to cover:
- All Turkish vowels: a, e, ı, i, o, ö, u, ü
- Vowel harmony patterns (front/back, rounded/unrounded)
- Consonants: b, c, ç, d, f, g, ğ, h, j, k, l, m, n, p, r, s, ş, t, v, y, z
- ğ in various positions (vowel lengthening)
- Sibilants: ş, ç, c, s, z
- Questions, exclamations, statements
- Numbers and compound words
- Formal and informal register
- Various sentence lengths (short to medium)

Write the file as a JSON array following the schema:

```json
{
  "language": "tr",
  "language_name": "Turkish",
  "description": "100 sentences for Turkish voice cloning, covering all phonemes and common patterns",
  "sentences": [
    {"id": 1, "text": "Güzel bir öğleden sonra, küçük çocuklar bahçede oynuyordu.", "category": "vowel_harmony"},
    ...
  ]
}
```

Use an LLM to draft sentences, then verify phoneme coverage:

```bash
uv run python -c "
import json
from misaki import espeak
from collections import Counter

g2p = espeak.EspeakG2P(language='tr')
with open('sentences/tr.json') as f:
    data = json.load(f)

all_phonemes = Counter()
for s in data['sentences']:
    ph, _ = g2p(s['text'])
    all_phonemes.update(ph)

print(f'Total sentences: {len(data[\"sentences\"])}')
print(f'Unique phonemes: {len(all_phonemes)}')
print(f'Top 20: {all_phonemes.most_common(20)}')
# Check for low-frequency phonemes
rare = [(p, c) for p, c in all_phonemes.items() if c < 3 and p.isalpha()]
if rare:
    print(f'WARNING: Low coverage phonemes: {rare}')
else:
    print('All phonemes have 3+ occurrences.')
"
```

- [ ] **Step 3: Commit**

```bash
git add sentences/
git commit -m "Add Turkish sentence script (100 sentences with phoneme coverage)"
```

---

### Task 7: Pipeline Integration (Quick Mode)

**Files:**
- Create: `backend/pipeline.py`
- Modify: `backend/app.py` (add process endpoints)

- [ ] **Step 1: Write backend/pipeline.py**

```python
"""Pipeline orchestration: voicepack extraction and fine-tuning."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def load_model_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(model_dir: Path, stage: str = "2nd") -> Path | None:
    """Find the latest checkpoint in the model's log directory."""
    log_dirs = sorted(model_dir.glob("StyleTTS2/logs/*/"))
    if not log_dirs:
        return None
    log_dir = log_dirs[-1]

    checkpoints = sorted(log_dir.glob(f"epoch_{stage}_*.pth"))
    if not checkpoints:
        checkpoints = sorted(log_dir.glob("epoch_1st_*.pth"))
    return checkpoints[-1] if checkpoints else None


def find_stage1_checkpoint(model_dir: Path) -> Path | None:
    log_dirs = sorted(model_dir.glob("StyleTTS2/logs/*/"))
    if not log_dirs:
        return None
    checkpoints = sorted(log_dirs[-1].glob("epoch_1st_*.pth"))
    return checkpoints[-1] if checkpoints else None


def extract_voicepack(
    model_dir: Path,
    recordings_dir: Path,
    output_path: Path,
) -> Path:
    """Run zero-shot voicepack extraction using user's recordings."""
    checkpoint = find_latest_checkpoint(model_dir)
    if checkpoint is None:
        raise FileNotFoundError("No checkpoint found in model directory")

    stage1_ckpt = find_stage1_checkpoint(model_dir)

    extract_script = model_dir / "scripts" / "extract_voicepack.py"
    if not extract_script.exists():
        raise FileNotFoundError(f"extract_voicepack.py not found at {extract_script}")

    cmd = [
        sys.executable,
        str(extract_script),
        "--model", str(checkpoint),
        "--audio-dir", str(recordings_dir),
        "--output", str(output_path),
    ]
    if stage1_ckpt and "2nd" in checkpoint.name:
        cmd.extend(["--style-encoder-model", str(stage1_ckpt)])

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(model_dir))
    if result.returncode != 0:
        raise RuntimeError(f"Voicepack extraction failed:\n{result.stderr}")

    return output_path


def generate_sample(
    model_dir: Path,
    voicepack_path: Path,
    output_path: Path,
    text: str = "Merhaba, ben sizin yeni sesinizim. Nasıl duyuluyorum?",
) -> Path:
    """Generate a test sample with the new voicepack."""
    checkpoint = find_latest_checkpoint(model_dir)
    if checkpoint is None:
        raise FileNotFoundError("No checkpoint found")

    test_script = model_dir / "scripts" / "test_inference.py"

    # We call the model directly rather than the full test script
    # to generate just one sentence
    cmd = [
        sys.executable, "-c", f"""
import sys
sys.path.insert(0, str({repr(str(model_dir / 'kokoro'))}))
import torch
import soundfile as sf
from kokoro import KModel
from misaki import espeak

g2p = espeak.EspeakG2P(language='tr')
phonemes, _ = g2p({repr(text)})
phonemes = phonemes.replace('\\u026b', 'l')
if len(phonemes) > 510:
    phonemes = phonemes[:510]

config_path = str({repr(str(model_dir / 'training' / 'config.json'))})

# Convert checkpoint
ckpt = torch.load({repr(str(checkpoint))}, map_location='cpu', weights_only=False)
net = ckpt['net']
weights = {{}}
for key in ['bert', 'bert_encoder', 'predictor', 'text_encoder', 'decoder']:
    if key in net:
        weights[key] = {{('module.' + k if not k.startswith('module.') else k): v for k, v in net[key].items()}}
converted = {repr(str(output_path.parent / 'converted_model.pth'))}
torch.save(weights, converted)

kmodel = KModel(repo_id='hexgrad/Kokoro-82M', config=config_path, model=converted)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
kmodel = kmodel.to(device).eval()

voice = torch.load({repr(str(voicepack_path))}, map_location='cpu', weights_only=True)
result = kmodel(phonemes, voice[len(phonemes)-1], 1.0, return_output=True)
sf.write({repr(str(output_path))}, result.audio, 24000)
print('OK')
"""
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Sample generation failed:\n{result.stderr}")

    return output_path
```

- [ ] **Step 2: Add process endpoints to backend/app.py**

Add these endpoints after the existing ones:

```python
from backend.pipeline import extract_voicepack, generate_sample, load_model_config

@app.post("/api/process/quick")
async def process_quick():
    if _session is None:
        raise HTTPException(404, "No active session")

    stats = _session.stats()
    if stats["recorded"] < 5:
        raise HTTPException(400, "Need at least 5 recordings")

    model_dir = Path(config["model_dir"])
    voicepack_path = _session.session_dir / "voicepack.pt"
    sample_path = _session.session_dir / "sample.wav"

    try:
        extract_voicepack(model_dir, _session.recordings_dir, voicepack_path)
        generate_sample(model_dir, voicepack_path, sample_path)
    except Exception as e:
        raise HTTPException(500, str(e))

    return {
        "message": "Voicepack created!",
        "voicepack": str(voicepack_path),
        "sample_url": f"/api/process/sample",
    }


@app.get("/api/process/sample")
def get_sample():
    if _session is None:
        raise HTTPException(404, "No active session")
    path = _session.session_dir / "sample.wav"
    if not path.exists():
        raise HTTPException(404, "No sample generated yet")
    return FileResponse(str(path), media_type="audio/wav")
```

- [ ] **Step 3: Test end-to-end manually**

```bash
uv run python -m backend.app
```

Open browser, create session, record a few sentences, click Quick Voicepack. Verify it extracts and plays a sample.

- [ ] **Step 4: Commit**

```bash
git add backend/pipeline.py backend/app.py
git commit -m "Add pipeline integration (quick voicepack extraction)"
```

---

### Task 8: Pipeline Integration (Full Mode)

**Files:**
- Modify: `backend/pipeline.py`
- Modify: `backend/app.py`

This task adds the fine-tuning pipeline. Since fine-tuning takes hours, it runs as a background process with status polling.

- [ ] **Step 1: Add fine-tune function to backend/pipeline.py**

```python
import json
import threading

_finetune_status = {"running": False, "epoch": 0, "step": 0, "loss": 0.0, "message": ""}


def get_finetune_status() -> dict:
    return dict(_finetune_status)


def run_finetune(
    model_dir: Path,
    session_dir: Path,
    recordings_dir: Path,
    language: str,
    sentences: list[dict],
    statuses: dict,
):
    """Run Stage 2 fine-tuning in a background thread."""
    global _finetune_status
    _finetune_status = {"running": True, "epoch": 0, "step": 0, "loss": 0.0, "message": "Preparing data..."}

    try:
        # Step 1: Run G2P on recorded sentences
        from misaki import espeak
        g2p = espeak.EspeakG2P(language=language)

        train_lines = []
        for s in sentences:
            sid = s["id"]
            if str(sid) not in statuses and sid not in statuses:
                continue
            status = statuses.get(str(sid), statuses.get(sid, "pending"))
            if status != "recorded":
                continue
            wav_path = recordings_dir / f"{sid:03d}.wav"
            if not wav_path.exists():
                continue
            phonemes, _ = g2p(s["text"])
            phonemes = phonemes.replace('ɫ', 'l')
            train_lines.append(f"{wav_path}|{phonemes}|0")

        if len(train_lines) < 5:
            _finetune_status = {"running": False, "message": "Not enough recordings (need 5+)"}
            return

        # Write train list (use all for training, skip validation for small datasets)
        train_list = session_dir / "train_list.txt"
        val_list = session_dir / "val_list.txt"
        split = max(1, len(train_lines) // 20)
        with open(val_list, "w") as f:
            f.write("\n".join(train_lines[:split]) + "\n")
        with open(train_list, "w") as f:
            f.write("\n".join(train_lines[split:]) + "\n")

        _finetune_status["message"] = f"Prepared {len(train_lines)} clips. Starting fine-tune..."

        # Step 2: TODO — invoke accelerate launch train_second.py
        # This requires creating a custom config pointing at the session's data
        # and the base model's Stage 1 checkpoint. For now, update status.
        _finetune_status = {
            "running": False,
            "message": f"Data prepared ({len(train_lines)} clips). Run fine-tuning manually — see session directory: {session_dir}",
        }

    except Exception as e:
        _finetune_status = {"running": False, "message": f"Error: {e}"}
```

- [ ] **Step 2: Add full-mode endpoints to backend/app.py**

```python
import threading
from backend.pipeline import run_finetune, get_finetune_status

@app.post("/api/process/full")
async def process_full():
    if _session is None:
        raise HTTPException(404, "No active session")

    model_dir = Path(config["model_dir"])
    thread = threading.Thread(
        target=run_finetune,
        args=(
            model_dir,
            _session.session_dir,
            _session.recordings_dir,
            _session.language,
            _session.sentences,
            _session._statuses,
        ),
        daemon=True,
    )
    thread.start()
    return {"message": "Fine-tuning started. Check status at /api/process/status"}


@app.get("/api/process/status")
def process_status():
    return get_finetune_status()
```

- [ ] **Step 3: Commit**

```bash
git add backend/pipeline.py backend/app.py
git commit -m "Add full fine-tune pipeline (data prep + background execution)"
```

---

### Task 9: End-to-End Test and Polish

**Files:**
- Modify: various (bug fixes found during testing)

- [ ] **Step 1: Start the server and test the full flow**

```bash
uv run python -m backend.app
```

Test in browser:
1. Create session with name
2. Record 5+ sentences using spacebar
3. Verify auto-advance works
4. Navigate with arrow keys
5. Re-record a sentence with R
6. Skip a sentence with S
7. Play back with P
8. Click Quick Voicepack
9. Listen to generated sample

- [ ] **Step 2: Fix any issues found**

Common issues to check:
- WebM to WAV conversion on the backend (browser sends WebM, backend needs to handle it)
- CORS if running on different ports
- Audio playback in browser
- Session persistence after browser refresh

- [ ] **Step 3: Commit fixes**

```bash
git add -A
git commit -m "End-to-end testing fixes and polish"
```

---

### Task 10: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README with full documentation**

Include:
- Project description and screenshot
- Quick start instructions
- Configuration
- How to add new languages (sentence script format)
- Quick vs Full mode explanation
- Requirements (mic access, trained Kokoro model)
- License

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Update documentation"
```
