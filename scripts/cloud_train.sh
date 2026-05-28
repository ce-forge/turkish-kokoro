#!/bin/bash
# Cloud Training Setup for Turkish Kokoro
# ========================================
# Paste this into a RunPod terminal (PyTorch template).
# Trains both stages and saves checkpoints for download.
#
# Prerequisites:
#   - Upload your dataset to the pod first:
#     scp -r dataset/audio/ runpod:/workspace/turkish-kokoro/dataset/audio/
#   - Or upload a tarball and extract it
#
# Usage:
#   bash scripts/cloud_train.sh

set -e

REPO_DIR="/workspace/turkish-kokoro"
BATCH_SIZE=12  # A40/A100 can handle 12 easily

echo "=== Turkish Kokoro Cloud Training ==="

# ── Step 1: Clone repo and submodules ────────────────────────────────────
if [ ! -d "$REPO_DIR" ]; then
    cd /workspace
    git clone --recurse-submodules https://github.com/YOUR_USERNAME/turkish-kokoro.git
fi
cd "$REPO_DIR"

# ── Step 2: Install system deps ──────────────────────────────────────────
apt-get update && apt-get install -y espeak-ng libsndfile1

# ── Step 3: Install Python deps ──────────────────────────────────────────
pip install uv
uv sync

# Install training deps
uv pip install accelerate munch tensorboard librosa pyyaml matplotlib einops einops-exts

# Build monotonic_align
cd StyleTTS2/monotonic_align
pip install .
cd "$REPO_DIR"

# ── Step 4: Convert base weights (if not already done) ───────────────────
if [ ! -f training/kokoro_base.pth ]; then
    echo "Converting Kokoro-82M weights..."
    uv run python scripts/prepare_training.py convert-weights
fi

# ── Step 5: Verify data ─────────────────────────────────────────────────
echo "Verifying training data..."
uv run python scripts/prepare_training.py verify

# ── Step 6: Update batch size for cloud GPU ──────────────────────────────
sed -i "s/^batch_size:.*/batch_size: $BATCH_SIZE/" configs/config_turkish_ft.yml
echo "Set batch_size to $BATCH_SIZE"

# ── Step 7: Configure accelerate ─────────────────────────────────────────
mkdir -p ~/.cache/huggingface/accelerate
cat > ~/.cache/huggingface/accelerate/default_config.yaml << 'ACCEL'
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: 'no'
num_processes: 1
ACCEL

# ── Step 8: Stage 1 Training ─────────────────────────────────────────────
echo ""
echo "=== Starting Stage 1 Training ==="
echo ""
cd StyleTTS2
accelerate launch train_first.py --config_path ../configs/config_turkish_ft.yml

# ── Step 9: Stage 2 Training ─────────────────────────────────────────────
echo ""
echo "=== Starting Stage 2 Training ==="
echo ""
accelerate launch train_second.py --config_path ../configs/config_turkish_ft.yml

# ── Step 10: Extract voicepacks ──────────────────────────────────────────
cd "$REPO_DIR"
echo ""
echo "=== Extracting Voicepacks ==="

LATEST_2ND=$(ls -t StyleTTS2/logs/kokoro-turkish/epoch_2nd_*.pth | head -1)
LATEST_1ST=$(ls -t StyleTTS2/logs/kokoro-turkish/epoch_1st_*.pth | head -1)

echo "Stage 2 checkpoint: $LATEST_2ND"
echo "Stage 1 checkpoint: $LATEST_1ST"

# Mazlum voicepack
uv run python scripts/extract_voicepack.py \
    --model "$LATEST_2ND" \
    --style-encoder-model "$LATEST_1ST" \
    --audio-dir dataset/audio/tm_mazlum \
    --output voices/tm_mazlum.pt

# Cankut voicepack (if data exists)
if [ -d "dataset/audio/tm_cankut" ]; then
    uv run python scripts/extract_voicepack.py \
        --model "$LATEST_2ND" \
        --style-encoder-model "$LATEST_1ST" \
        --audio-dir dataset/audio/tm_cankut \
        --output voices/tm_cankut.pt
fi

# ── Step 11: Generate test samples ───────────────────────────────────────
echo ""
echo "=== Generating Test Samples ==="

uv run python scripts/test_inference.py \
    --checkpoint "$LATEST_2ND" \
    --voicepack voices/tm_mazlum.pt \
    --output-dir test_output/cloud_mazlum

if [ -f "voices/tm_cankut.pt" ]; then
    uv run python scripts/test_inference.py \
        --checkpoint "$LATEST_2ND" \
        --voicepack voices/tm_cankut.pt \
        --output-dir test_output/cloud_cankut
fi

# ── Done ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Training complete!"
echo ""
echo "Download these files:"
echo "  Checkpoints:"
echo "    $LATEST_1ST"
echo "    $LATEST_2ND"
echo "  Voicepacks:"
echo "    voices/tm_mazlum.pt"
[ -f "voices/tm_cankut.pt" ] && echo "    voices/tm_cankut.pt"
echo "  Test audio:"
echo "    test_output/cloud_mazlum/"
[ -d "test_output/cloud_cankut" ] && echo "    test_output/cloud_cankut/"
echo ""
echo "Total size to download: ~2.5GB (checkpoints + voicepacks + audio)"
echo "============================================================"
