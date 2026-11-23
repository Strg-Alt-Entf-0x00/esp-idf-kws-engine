"""PyTorch Training Pipeline for ESP32-P4 TC-ResNet Streaming KWS Models."""

import argparse
import random
import wave
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from frontend import extract_features, SAMPLE_RATE_HZ
from model import TCResNet
from export_espm import export_kws_to_espm


def load_wav_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        pcm = np.frombuffer(frames, dtype=np.int16)
        if wf.getnchannels() > 1:
            pcm = pcm[::wf.getnchannels()]
        return pcm


def get_peak_energy_window(pcm: np.ndarray, target_samples: int) -> int:
    if len(pcm) <= target_samples:
        return 0
    chunks = len(pcm) // 160
    w_chunks = target_samples // 160
    if chunks <= w_chunks:
        return 0
    energies = [float(np.sum(pcm[i * 160 : (i + 1) * 160] ** 2)) for i in range(chunks)]
    best_chunk = 0
    max_e = -1.0
    for i in range(len(energies) - w_chunks + 1):
        e = sum(energies[i : i + w_chunks])
        if e > max_e:
            max_e = e
            best_chunk = i
    return best_chunk * 160


class KWSDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], augment: bool = True, repeats: int = 1):
        self.samples = samples * (repeats if augment else 1)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        required_samples = (64 - 1) * 160 + 480  # 10,560 samples = 640 ms

        # 1. Background / Silence synthesis
        if label == 0:
            if self.augment and random.random() < 0.35:
                if random.random() < 0.5:
                    pcm = np.zeros(required_samples, dtype=np.float32)
                else:
                    pcm = np.random.normal(0, random.uniform(1.0, 15.0), required_samples).astype(np.float32)
            else:
                pcm = load_wav_pcm(path).astype(np.float32)
                if len(pcm) > required_samples:
                    max_s = len(pcm) - required_samples
                    start = random.randint(0, max_s) if self.augment else 0
                    pcm = pcm[start : start + required_samples]
                elif len(pcm) < required_samples:
                    pcm = np.pad(pcm, (0, required_samples - len(pcm)), mode="constant")
        # 2. Target Keyword (Active energy window)
        else:
            pcm = load_wav_pcm(path).astype(np.float32)
            if len(pcm) > required_samples:
                best_start = get_peak_energy_window(pcm, required_samples)
                if self.augment:
                    jitter = random.randint(-800, 800)
                    start = max(0, min(len(pcm) - required_samples, best_start + jitter))
                else:
                    start = best_start
                pcm = pcm[start : start + required_samples]
            elif len(pcm) < required_samples:
                pcm = np.pad(pcm, (0, required_samples - len(pcm)), mode="constant")

            if self.augment:
                # Random gain: 0.8 to 1.2
                gain = random.uniform(0.8, 1.2)
                pcm = pcm * gain

                # Faint additive noise
                noise = np.random.normal(0, random.uniform(2.0, 25.0), len(pcm))
                pcm = pcm + noise

        features = extract_features(pcm, num_frames=64)
        return features, label


def train_kws(
    data_dir: Path,
    output_espm: Path,
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 0.002,
    seed: int = 42
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Scan classes from subdirectories: data_dir/class_name/*.wav
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not classes:
        raise ValueError(f"No class folders found in '{data_dir}'. Expected e.g. '0_background/', '1_target/'")

    all_samples = []
    for class_idx, class_name in enumerate(classes):
        wavs = list((data_dir / class_name).glob("*.wav"))
        for w in wavs:
            all_samples.append((w, class_idx))

    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.85)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f"[KWS Trainer] Found {len(all_samples)} samples across {len(classes)} classes: {classes}")
    print(f"[KWS Trainer] Training split: {len(train_samples)}, Validation split: {len(val_samples)}")

    train_ds = KWSDataset(train_samples, augment=True, repeats=10)
    val_ds = KWSDataset(val_samples, augment=False, repeats=1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TCResNet(num_classes=len(classes), in_bins=40)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for feats, labels in train_loader:
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        scheduler.step()
        train_acc = correct / max(total, 1)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                logits = model(feats)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / max(val_total, 1)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {total_loss/total:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\n[KWS Trainer] Training complete! Best validation accuracy: {best_acc*100:.1f}%")
    export_kws_to_espm(model, classes, output_espm, model_name=output_espm.stem)


def main():
    parser = argparse.ArgumentParser(description="Train TC-ResNet for ESP32-P4 KWS Engine")
    parser.add_argument("--data", required=True, type=Path, help="Dataset directory containing class folders")
    parser.add_argument("--output", required=True, type=Path, help="Output .espm model file")
    parser.add_argument("--epochs", type=int, default=25, help="Epoch count")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    train_kws(args.data, args.output, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
