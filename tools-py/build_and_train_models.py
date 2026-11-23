"""End-to-End Autonomous Model Training & Dataset Generation for ESP-IDF KWS Engine."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent
TOOLS_PY = ROOT_DIR / "tools-py"
SYNTH_SCRIPT = TOOLS_PY / "synth_voxcpm.py"
TRAIN_SCRIPT = TOOLS_PY / "train.py"

DEFAULT_DATA_DIR = ROOT_DIR / ".data"
DEFAULT_MODELS_DIR = ROOT_DIR / "models" / "espm"
DEFAULT_TEST_AUDIO_DIR = ROOT_DIR / ".data" / "test_audio"


def run_py(python_bin: str, args):
    cmd = [python_bin] + [str(a) for a in args]
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="End-to-end dataset synthesis, training, and ESPM export")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Working directory for datasets (default: .data)")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Output directory for .espm models")
    parser.add_argument("--test-audio-dir", type=Path, default=DEFAULT_TEST_AUDIO_DIR, help="Output directory for benchmark WAVs")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python binary for sub-processes")
    parser.add_argument("--voxcpm-dir", type=Path, default=None, help="Optional directory to voxcpm-synthesis")
    args = parser.parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.test_audio_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. Wake Word Dataset & Model ("Hey Espira")
    # =========================================================================
    wake_data = args.data_dir / "wake_dataset"
    wake_bg = wake_data / "0_background"
    wake_tgt = wake_data / "1_hey_espira"

    print("=== [1/4] Synthesizing Wake Word Dataset ===")
    synth_wake = [SYNTH_SCRIPT, "--phrase", "Hey Espira", "--output", wake_tgt, "--count", "8"]
    if args.voxcpm_dir:
        synth_wake.extend(["--voxcpm-dir", args.voxcpm_dir])
    run_py(args.python, synth_wake)

    synth_bg = [SYNTH_SCRIPT, "--phrase", "Computer hallo guten morgen wie ist das wetter", "--output", wake_bg, "--count", "8"]
    if args.voxcpm_dir:
        synth_bg.extend(["--voxcpm-dir", args.voxcpm_dir])
    run_py(args.python, synth_bg)

    wake_espm = args.models_dir / "wake_ghost.espm"
    print(f"\n=== [2/4] Training Wake Word Model -> {wake_espm} ===")
    run_py(args.python, [TRAIN_SCRIPT, "--data", wake_data, "--output", wake_espm, "--epochs", "20", "--batch-size", "8"])

    # =========================================================================
    # 2. Command Word Dataset & Model ("Licht an", "Licht aus", "Stop")
    # =========================================================================
    cmd_data = args.data_dir / "cmd_dataset"
    cmd_bg = cmd_data / "0_background"
    cmd_c1 = cmd_data / "1_licht_an"
    cmd_c2 = cmd_data / "2_licht_aus"
    cmd_c3 = cmd_data / "3_stop"

    print("\n=== [3/4] Synthesizing Command Words Dataset ===")
    for phrase, out_dir in [
        ("hintergrund pause ruhe", cmd_bg),
        ("Licht an", cmd_c1),
        ("Licht aus", cmd_c2),
        ("Stop", cmd_c3),
    ]:
        scmd = [SYNTH_SCRIPT, "--phrase", phrase, "--output", out_dir, "--count", "6"]
        if args.voxcpm_dir:
            scmd.extend(["--voxcpm-dir", args.voxcpm_dir])
        run_py(args.python, scmd)

    cmd_espm = args.models_dir / "cmd_smarthome.espm"
    print(f"\n=== [4/4] Training Command Model -> {cmd_espm} ===")
    run_py(args.python, [TRAIN_SCRIPT, "--data", cmd_data, "--output", cmd_espm, "--epochs", "20", "--batch-size", "8"])

    # Copy benchmark WAVs for testing
    print("\n=== Preparing Benchmark Test WAVs ===")
    sample_wake = next(wake_tgt.glob("*.wav"), None)
    if sample_wake:
        shutil.copy2(sample_wake, args.test_audio_dir / "test_wake_espira.wav")

    sample_cmd1 = next(cmd_c1.glob("*.wav"), None)
    if sample_cmd1:
        shutil.copy2(sample_cmd1, args.test_audio_dir / "test_cmd_licht_an.wav")

    sample_cmd3 = next(cmd_c3.glob("*.wav"), None)
    if sample_cmd3:
        shutil.copy2(sample_cmd3, args.test_audio_dir / "test_cmd_stop.wav")

    sample_bg = next(wake_bg.glob("*.wav"), None)
    if sample_bg:
        shutil.copy2(sample_bg, args.test_audio_dir / "test_background.wav")

    print(f"\n[DONE] Models generated in '{args.models_dir}'")
    print(f"[DONE] Benchmark WAVs generated in '{args.test_audio_dir}'")


if __name__ == "__main__":
    main()
