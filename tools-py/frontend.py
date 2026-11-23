"""Bit-Exact Audio Frontend matching C implementation in kws_frontend.c."""

import math
import numpy as np
import torch

SAMPLE_RATE_HZ = 16000
HOP_SAMPLES = 160
WIN_SAMPLES = 480
FFT_SIZE = 512
NUM_MELS = 40
PREEMPHASIS_COEFF = 0.97
LOG_EPSILON = 1e-5
MEL_LOW_HZ = 80.0
MEL_HIGH_HZ = 7600.0


def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def get_mel_filterbanks() -> np.ndarray:
    """Builds triangular mel filterbank matrix of shape (NUM_MELS, FFT_SIZE // 2 + 1)."""
    mel_min = hz_to_mel(MEL_LOW_HZ)
    mel_max = hz_to_mel(MEL_HIGH_HZ)
    mel_points = np.linspace(mel_min, mel_max, NUM_MELS + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])

    hz_per_bin = SAMPLE_RATE_HZ / FFT_SIZE
    bin_points = np.floor((FFT_SIZE + 1) * hz_points / SAMPLE_RATE_HZ).astype(int)

    fbank = np.zeros((NUM_MELS, FFT_SIZE // 2 + 1), dtype=np.float32)
    for m in range(1, NUM_MELS + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m > f_m_minus:
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus > f_m:
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    return fbank


def extract_features(audio_pcm: np.ndarray, num_frames: int = 64) -> torch.Tensor:
    """Extracts (NUM_MELS, num_frames) log-mel spectrogram from 16kHz mono PCM audio.
    
    Args:
        audio_pcm: 1D float32 or int16 array of audio samples.
        num_frames: fixed time length (default 64 frames = 640 ms).
    """
    if audio_pcm.dtype == np.int16:
        audio_pcm = audio_pcm.astype(np.float32)

    # Pre-emphasis: y[n] = x[n] - alpha * x[n-1]
    emphasized = np.append(audio_pcm[0], audio_pcm[1:] - PREEMPHASIS_COEFF * audio_pcm[:-1])

    # Periodic Hann window
    hann_window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(WIN_SAMPLES) / WIN_SAMPLES)
    fbank = get_mel_filterbanks()

    total_samples = len(emphasized)
    required_samples = (num_frames - 1) * HOP_SAMPLES + WIN_SAMPLES

    if total_samples < required_samples:
        pad_len = required_samples - total_samples
        emphasized = np.pad(emphasized, (0, pad_len), mode="constant")

    mel_spectrogram = np.zeros((NUM_MELS, num_frames), dtype=np.float32)
    offset = (FFT_SIZE - WIN_SAMPLES) // 2

    for frame_idx in range(num_frames):
        start = frame_idx * HOP_SAMPLES
        windowed = emphasized[start : start + WIN_SAMPLES] * hann_window

        fft_input = np.zeros(FFT_SIZE, dtype=np.float32)
        fft_input[offset : offset + WIN_SAMPLES] = windowed

        spectrum = np.fft.rfft(fft_input)
        power_spectrum = np.abs(spectrum) ** 2

        mel_energies = np.dot(fbank, power_spectrum)
        log_mel = np.log(mel_energies + LOG_EPSILON)
        mel_spectrogram[:, frame_idx] = log_mel

    return torch.from_numpy(mel_spectrogram)
