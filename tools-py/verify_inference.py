import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import wave

sys.path.insert(0, str(Path(__file__).parent))
from frontend import extract_features, get_mel_filterbanks, WIN_SAMPLES, HOP_SAMPLES, FFT_SIZE, PREEMPHASIS_COEFF, LOG_EPSILON
from export_espm import INDEX_HEADER_STRUCT, TENSOR_RECORD_STRUCT, HEADER_STRUCT, SECTION_STRUCT

def load_wav(p):
    with wave.open(str(p), "rb") as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)

def load_espm_tensors(path):
    with open(path, "rb") as f:
        buf = f.read()

    hdr = HEADER_STRUCT.unpack_from(buf, 0)
    num_sections = hdr[4]
    sec_offset = HEADER_STRUCT.size

    sections = {}
    for i in range(num_sections):
        sec = SECTION_STRUCT.unpack_from(buf, sec_offset + i * SECTION_STRUCT.size)
        sec_type = sec[0]
        s_off = sec[2]
        s_sz = sec[3]
        sections[sec_type] = (s_off, s_sz)

    idx_off, idx_sz = sections[6]
    dat_off, dat_sz = sections[7]

    idx_hdr = INDEX_HEADER_STRUCT.unpack_from(buf, idx_off)
    tensor_count = idx_hdr[5]

    rec_offset = idx_off + INDEX_HEADER_STRUCT.size
    tensors = {}
    for i in range(tensor_count):
        rec = TENSOR_RECORD_STRUCT.unpack_from(buf, rec_offset + i * TENSOR_RECORD_STRUCT.size)
        name = rec[22].decode("utf-8").rstrip("\x00")
        shape_len = rec[3]
        shape = rec[7:7+shape_len]
        data_offset = rec[20]
        data_len = rec[21]
        raw_data = buf[dat_off + data_offset : dat_off + data_offset + data_len]
        arr = np.frombuffer(raw_data, dtype=np.float32).reshape(shape)
        tensors[name] = torch.from_numpy(arr.copy())
    return tensors

def conv1d_forward(x, weight, bias, padding=1, dilation=1, relu=True):
    out = F.conv1d(x, weight, bias, padding=padding, dilation=dilation)
    return F.relu(out) if relu else out

def run_forward(tensors, feat_40x64):
    x = feat_40x64
    if x.ndim == 2:
        x = x.unsqueeze(0)

    # Conv0
    c0 = conv1d_forward(x, tensors["conv0.weight"], tensors["conv0.bias"], padding=1, dilation=1, relu=True)

    # Block 1
    b1_c1 = conv1d_forward(c0, tensors["block1.conv1.weight"], tensors["block1.conv1.bias"], padding=1, dilation=1, relu=True)
    b1_c2 = conv1d_forward(b1_c1, tensors["block1.conv2.weight"], tensors["block1.conv2.bias"], padding=1, dilation=1, relu=False)
    b1_out = F.relu(b1_c2 + c0)

    # Block 2
    b2_skip = conv1d_forward(b1_out, tensors["block2.skip.weight"], tensors["block2.skip.bias"], padding=0, dilation=1, relu=False)
    b2_c1 = conv1d_forward(b1_out, tensors["block2.conv1.weight"], tensors["block2.conv1.bias"], padding=2, dilation=2, relu=True)
    b2_c2 = conv1d_forward(b2_c1, tensors["block2.conv2.weight"], tensors["block2.conv2.bias"], padding=1, dilation=1, relu=False)
    b2_out = F.relu(b2_c2 + b2_skip)

    # Block 3
    b3_c1 = conv1d_forward(b2_out, tensors["block3.conv1.weight"], tensors["block3.conv1.bias"], padding=2, dilation=2, relu=True)
    b3_c2 = conv1d_forward(b3_c1, tensors["block3.conv2.weight"], tensors["block3.conv2.bias"], padding=1, dilation=1, relu=False)
    b3_out = F.relu(b3_c2 + b2_out)

    # Global Average Pooling
    pooled = b3_out.mean(dim=-1)

    # Classifier
    logits = F.linear(pooled, tensors["classifier.weight"], tensors["classifier.bias"])
    probs = F.softmax(logits, dim=-1)
    return probs.squeeze(0).numpy()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify KWS model inference (Python reference vs ESPM)")
    root_dir = Path(__file__).resolve().parent.parent
    default_model = root_dir / "models" / "espm" / "wake_ghost.espm"
    default_audio_dir = root_dir / ".data" / "test_audio"

    parser.add_argument("--model", type=Path, default=default_model, help="Path to .espm model file")
    parser.add_argument("--test-audio-dir", type=Path, default=default_audio_dir, help="Directory containing test WAV files")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"[Error] Model file '{args.model}' not found. Please provide via --model.")
        return 1

    tensors = load_espm_tensors(str(args.model))
    print(f"Loaded {len(tensors)} tensors successfully from '{args.model.name}'.")

    # 1. Test All Zeros
    zeros = torch.zeros(1, 40, 64)
    p_zeros = run_forward(tensors, zeros)
    print(f"Prediction on ALL ZEROS: [0_bg={p_zeros[0]:.4f}, 1_wake={p_zeros[1]:.4f}]")

    # 2. Test Silence (log 1e-5 = -11.5)
    silence = torch.full((1, 40, 64), -11.5129)
    p_sil = run_forward(tensors, silence)
    print(f"Prediction on SILENCE (-11.5): [0_bg={p_sil[0]:.4f}, 1_wake={p_sil[1]:.4f}]")

    # 3. Test Audio files
    if args.test_audio_dir.exists():
        wav_files = sorted(args.test_audio_dir.glob("*.wav"))
        for wav_path in wav_files:
            pcm = load_wav(wav_path)
            probs = run_forward(tensors, extract_features(pcm, 64))
            prob_str = ", ".join(f"c{i}={p:.4f}" for i, p in enumerate(probs))
            print(f"Prediction on '{wav_path.name}': [{prob_str}]")
    else:
        print(f"[Notice] Test audio directory '{args.test_audio_dir}' not found. Skipping WAV evaluation.")
    return 0

if __name__ == "__main__":
    main()
