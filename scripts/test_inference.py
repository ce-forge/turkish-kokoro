#!/usr/bin/env python3
"""
Turkish Kokoro: Test Inference
===============================
Converts a StyleTTS2 checkpoint to Kokoro KModel format and runs inference
on a set of Turkish test sentences to verify model quality.

Usage:
    # From a raw StyleTTS2 checkpoint (will convert first):
    uv run python scripts/test_inference.py \
        --checkpoint StyleTTS2/logs/kokoro-turkish/epoch_2nd_00014.pth \
        --voicepack voices/tm_mazlum.pt \
        --output-dir test_output/

    # From an already-converted Kokoro model:
    uv run python scripts/test_inference.py \
        --model models/kokoro-turkish.pth \
        --voicepack voices/tm_mazlum.pt \
        --output-dir test_output/
"""

import argparse
import sys
from pathlib import Path

# Add kokoro submodule to path
_repo_root = Path(__file__).resolve().parents[1]
_kokoro_submodule = _repo_root / "kokoro"
if _kokoro_submodule.exists() and str(_kokoro_submodule) not in sys.path:
    sys.path.insert(0, str(_kokoro_submodule))

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
    """Convert a StyleTTS2 checkpoint to Kokoro KModel format.

    Extracts the 5 core components (bert, bert_encoder, predictor,
    text_encoder, decoder) and adds the 'module.' prefix to all keys
    so they match the DataParallel format that KModel expects.

    Args:
        checkpoint_path: Path to the StyleTTS2 .pth checkpoint.
        output_path: Where to save the converted Kokoro model weights.

    Returns:
        The output_path string.
    """
    import torch

    print(f"Converting checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net = ckpt["net"]

    def ensure_module_prefix(state_dict):
        """Ensure all keys have the 'module.' prefix (DataParallel format)."""
        return {
            ("module." + k if not k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }

    kokoro_weights = {}
    component_keys = ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"]
    for key in component_keys:
        if key in net:
            kokoro_weights[key] = ensure_module_prefix(net[key])
            n_params = sum(v.numel() for v in net[key].values())
            print(f"  Extracted {key}: {len(net[key])} tensors, {n_params:,} parameters")
        else:
            print(f"  WARNING: '{key}' not found in checkpoint")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(kokoro_weights, output_path)
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Saved converted model: {output_path} ({size_mb:.1f} MB)")

    epoch = ckpt.get("epoch", "?")
    print(f"  Source checkpoint epoch: {epoch}")

    return output_path


def run_inference(
    model_path: str,
    voicepack_path: str,
    config_path: str,
    output_dir: str,
    device: str = "auto",
) -> None:
    """Load a Kokoro model and generate audio for all Turkish test sentences.

    Args:
        model_path: Path to Kokoro-format model weights (.pth).
        voicepack_path: Path to voicepack (.pt).
        config_path: Path to model config JSON.
        output_dir: Directory where output .wav files will be written.
        device: 'auto', 'cpu', or 'cuda'.
    """
    import numpy as np
    import soundfile as sf
    import torch
    from kokoro import KModel, KPipeline

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model: {model_path}")
    print(f"Loading config: {config_path}")
    kmodel = KModel(repo_id="hexgrad/Kokoro-82M", config=config_path, model=model_path)
    kmodel = kmodel.to(device).eval()

    # Build pipeline with Turkish language code
    pipeline = KPipeline(lang_code="t", repo_id="hexgrad/Kokoro-82M", model=kmodel)

    # Load voicepack
    print(f"Loading voicepack: {voicepack_path}")
    voice = torch.load(voicepack_path, map_location="cpu", weights_only=True)
    print(f"  Voicepack shape: {tuple(voice.shape)}")

    # Prepare output directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {len(TEST_SENTENCES)} test sentences...\n")
    for i, text in enumerate(TEST_SENTENCES):
        print(f"  [{i + 1}/{len(TEST_SENTENCES)}] {text[:60]}...")

        generator = pipeline(text, voice=voice, speed=1)
        all_audio = []
        for gs, ps, audio in generator:
            all_audio.append(audio)

        if all_audio:
            combined = np.concatenate(all_audio)
            wav_path = out / f"test_{i + 1:02d}.wav"
            sf.write(str(wav_path), combined, 24000)
            duration = len(combined) / 24000
            print(f"         -> {wav_path} ({duration:.2f}s)")
        else:
            print(f"         -> WARNING: No audio generated for sentence {i + 1}")

    print(f"\nDone! Output files are in: {out}/")


def main():
    parser = argparse.ArgumentParser(
        description="Turkish Kokoro inference test: convert checkpoint and generate test audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a StyleTTS2 checkpoint and run inference:
  uv run python scripts/test_inference.py \\
      --checkpoint StyleTTS2/logs/kokoro-turkish/epoch_2nd_00014.pth \\
      --voicepack voices/tm_mazlum.pt \\
      --output-dir test_output/

  # Use an already-converted model:
  uv run python scripts/test_inference.py \\
      --model models/kokoro-turkish.pth \\
      --voicepack voices/tm_mazlum.pt \\
      --output-dir test_output/

  # Force CPU:
  uv run python scripts/test_inference.py \\
      --checkpoint StyleTTS2/logs/kokoro-turkish/epoch_2nd_00014.pth \\
      --voicepack voices/tm_mazlum.pt \\
      --device cpu
""",
    )

    # Mutually exclusive: --checkpoint (raw StyleTTS2) or --model (already converted)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--checkpoint",
        help="Path to a raw StyleTTS2 checkpoint (.pth). Will be converted to "
        "Kokoro format before inference.",
    )
    source_group.add_argument(
        "--model",
        help="Path to an already-converted Kokoro model (.pth).",
    )

    parser.add_argument(
        "--voicepack",
        required=True,
        help="Path to voicepack file (.pt).",
    )
    parser.add_argument(
        "--config",
        default="training/config.json",
        help="Path to model config JSON (default: training/config.json).",
    )
    parser.add_argument(
        "--output-dir",
        default="test_output/",
        help="Directory for output WAV files (default: test_output/).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on: auto (default), cpu, or cuda.",
    )

    args = parser.parse_args()

    # Determine model path — convert if needed
    if args.checkpoint:
        # Convert StyleTTS2 checkpoint to Kokoro format
        converted_name = Path(args.checkpoint).stem + "_kokoro.pth"
        converted_path = str(Path(args.output_dir) / converted_name)
        model_path = convert_checkpoint(args.checkpoint, converted_path)
    else:
        model_path = args.model

    # Run inference
    run_inference(
        model_path=model_path,
        voicepack_path=args.voicepack,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
