"""Self-Test Script for KWS TC-ResNet Architecture and ESPM Exporter."""

import sys
from pathlib import Path
import torch

from model import TCResNet
from export_espm import export_kws_to_espm, ESPM_MAGIC_LE, ESPM_VERSION, MODEL_TYPE_KWS
import struct


def run_pipeline_test():
    print("[Self-Test] 1. Initializing PyTorch TC-ResNet model...")
    model = TCResNet(num_classes=3, in_bins=40)
    model.eval()

    # Create dummy batch (Batch=2, Bins=40, Time=64)
    dummy_input = torch.randn(2, 40, 64)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (2, 3), f"Unexpected output shape: {out.shape}"
    print(f"[Self-Test] Forward pass successful. Output shape: {out.shape}")

    # Export to ESPM
    test_espm_path = Path("test_output/wake_test.espm")
    labels = ["background", "wake_ghost", "impostor"]
    print(f"[Self-Test] 2. Exporting to ESPM v3: '{test_espm_path}'...")
    export_kws_to_espm(model, labels, test_espm_path, model_name="wake_ghost_test")

    # Read back and validate binary header
    print("[Self-Test] 3. Validating exported binary ESPM container...")
    with open(test_espm_path, "rb") as f:
        data = f.read()

    assert len(data) > 128, "File too small"
    magic, version, m_type, q_type, n_sec = struct.unpack("<IHHHH", data[:12])
    assert magic == ESPM_MAGIC_LE, f"Magic mismatch: 0x{magic:08X}"
    assert version == ESPM_VERSION, f"Version mismatch: {version}"
    assert m_type == MODEL_TYPE_KWS, f"Model type mismatch: {m_type}"

    print(f"[Self-Test] Magic: 0x{magic:08X} ('ESPM'), Version: {version}, ModelType: {m_type} (KWS), Sections: {n_sec}")
    print(f"[Self-Test] Total file size: {len(data)} bytes ({len(data) / 1024:.2f} KB)")
    print("[Self-Test] ALL CHECKS PASSED SUCCESSFULLY!")

    # Clean up
    test_espm_path.unlink()
    test_espm_path.parent.rmdir()


if __name__ == "__main__":
    run_pipeline_test()
