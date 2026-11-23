"""Dataset Generation Tool using VoxCPM2 for Voice-Cloning and Impostor Synthesis."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

def find_default_voxcpm_dir() -> Optional[Path]:
    env_dir = os.environ.get("VOXCPM_DIR")
    if env_dir and Path(env_dir).is_dir():
        return Path(env_dir)
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root.parent / "esp32-p4-secure-voice-system" / "tools" / "pc-tool" / "tools" / "voxcpm-synthesis",
        repo_root.parent / "command-wakeword-system" / "voxcpm-synthesis",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None

IMPOSTOR_PROFILES = [
    "deep, resonant male voice",
    "bright, clear female voice",
    "calm, slow male voice",
    "energetic, high-pitched female voice",
    "elderly, raspy voice",
    "young, soft spoken voice",
    "nasal, fast-paced voice",
    "warm, storytelling radio host voice"
]


def synthesize_phrase_variants(
    phrase: str,
    output_dir: Path,
    reference_audio: Optional[Path] = None,
    num_variants: int = 20,
    voxcpm_dir: Optional[Path] = None
) -> List[Path]:
    """Generates synthetic audio files for a phrase.
    
    If reference_audio is provided -> Voice Cloning mode (UNIQUE speaker model).
    If reference_audio is None -> Multi-Voice Impostor mode (Negative classes).
    """
    if voxcpm_dir is None:
        voxcpm_dir = find_default_voxcpm_dir()

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    print(f"\n[VoxCPM2 Synth] Synthesizing '{phrase}' ({'Voice-Clone: ' + str(reference_audio) if reference_audio else 'Multi-Voice Impostors'})...")

    script_path = (voxcpm_dir / "synthesize_dataset.py") if voxcpm_dir else None
    if not script_path or not script_path.exists():
        print(f"[Notice] VoxCPM2 synthesis script not found. Creating synthetic fallback WAVs.")
        import wave
        import numpy as np
        for i in range(num_variants):
            wav_path = output_dir / f"synth_{i:04d}.wav"
            t = np.linspace(0, 1.0, 16000, False)
            freq = 300 + (i * 45) % 600
            sig = (0.2 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(sig.tobytes())
            generated_files.append(wav_path)
        return generated_files

    # Prepare project manifest conforming to VoxCPM2 schema
    phrase_id = str(uuid.uuid4())
    manifest_data = {
        "schemaVersion": 1,
        "projectId": str(uuid.uuid4()),
        "phrases": [{
            "id": phrase_id,
            "text": phrase,
            "language": "de",
            "role": "wake"
        }],
        "samples": []
    }
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        project_json = tmp_path / "project.json"
        project_json.write_text(json.dumps(manifest_data), encoding="utf-8")
        
        synth_output_dir = tmp_path / "synth_output"
        
        # Calculate samples per control profile
        if reference_audio and reference_audio.exists():
            controls = []
            samples_per_phrase = max(1, num_variants)
        else:
            controls = IMPOSTOR_PROFILES[:min(4, len(IMPOSTOR_PROFILES))]
            samples_per_phrase = max(1, (num_variants + len(controls) - 1) // len(controls))

        cmd = [
            "uv", "run", "--project", str(voxcpm_dir), "python", str(script_path),
            "--project", str(project_json),
            "--output", str(synth_output_dir),
            "--samples-per-phrase", str(samples_per_phrase),
            "--device", "auto"
        ]

        if reference_audio and reference_audio.exists():
            cmd.extend([
                "--reference-audio", str(reference_audio.resolve()),
                "--confirm-reference-permission"
            ])
        else:
            for profile in controls:
                cmd.extend(["--control", profile])

        try:
            print(f"[VoxCPM2 Synth] Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            samples_source = synth_output_dir / "samples"
            if samples_source.exists():
                for wav in samples_source.glob("*.wav"):
                    dest = output_dir / f"voxcpm_{wav.name}"
                    shutil.copy2(wav, dest)
                    generated_files.append(dest)
            print(f"[VoxCPM2 Synth] Generated {len(generated_files)} WAV files into {output_dir}")
        except Exception as e:
            print(f"[Error] Failed to run VoxCPM2 synthesis: {e}")

    return generated_files


def main():
    parser = argparse.ArgumentParser(description="VoxCPM2 KWS Dataset Generator")
    parser.add_argument("--phrase", required=True, help="Wake or Command phrase to synthesize")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for generated WAVs")
    parser.add_argument("--reference-audio", type=Path, default=None, help="Reference audio of your voice for cloning (UNIQUE mode)")
    parser.add_argument("--count", type=int, default=20, help="Number of audio variations to produce")
    parser.add_argument("--voxcpm-dir", type=Path, default=None, help="Directory containing voxcpm-synthesis")
    args = parser.parse_args()

    synthesize_phrase_variants(
        phrase=args.phrase,
        output_dir=args.output,
        reference_audio=args.reference_audio,
        num_variants=args.count,
        voxcpm_dir=args.voxcpm_dir
    )


if __name__ == "__main__":
    main()
