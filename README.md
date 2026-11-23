# ESP-IDF KWS Engine (`esp-idf-kws-engine`)

A lightweight, high-performance, streaming Keyword Spotting (KWS) and Voice Command Engine for ESP-IDF.

`esp-idf-kws-engine` delivers real-time voice trigger and command recognition directly on the microcontroller with zero third-party ML dependencies (no ESP-DL, no TensorFlow Lite Micro). Built entirely in standard C, it achieves deterministic, low-latency audio inference on ESP32-P4 and ESP32-S3 targets.

## Supported Configuration

- **Target Microcontrollers:** ESP32-P4 (RISC-V Dual-Core @ 360 MHz / 400 MHz), ESP32-S3 (Xtensa Dual-Core @ 240 MHz)
- **Framework:** ESP-IDF v6.0.2 or later
- **Audio Format:** 16 kHz, 16-bit signed mono PCM (or normalized Float32 in `[-1.0, 1.0]`)
- **Hop Size:** 160 samples per frame (10 ms hop window)
- **Sliding Context Window:** 64 frames (640 ms temporal context)
- **Dependencies:** `esp-dsp` (for fast FFT calculation) and [`esp-idf-model-manager`](https://github.com/Strg-Alt-Entf-0x00/esp-idf-model-manager) (for ESPM container loading)

## Key Features

- **100% Native C Inference:** Zero third-party machine learning runtime overhead. Hand-optimized, cache-linear 1D temporal convolution and residual operations running directly on the hardware FPU.
- **Streaming TC-ResNet Architecture:** Temporal Convolutional ResNet with decoupled residual connections. Processes audio frame-by-frame (10 ms hops) with a continuous circular feature buffer.
- **ESPM Binary Container Format:** Fully integrated with `esp-idf-model-manager`. Models are compact, portable binary files with CRC32 integrity verification (`MODEL_TYPE_KWS = 2`) and folded BatchNorm weights.
- **Decoupled Wake & Command Models:** Wake-word (`wake_*.espm`) and command-word (`cmd_*.espm`) engines are independent. New voice commands can be added or updated dynamically without retraining or touching the wake-word model.
- **Speaker-Dependent (UNIQUE) & Speaker-Independent (GENERAL) Modes:**
  - **UNIQUE Mode:** Trained on the user's specific voice (via VoxCPM2 voice cloning) against acoustic negative impostors, eliminating the need for a separate biometrics engine.
  - **GENERAL Mode:** Trained across multiple speakers for universal wake-up and command activation.
- **Deterministic Memory Footprint:** Zero heap allocations during steady-state streaming inference. All internal buffers are pre-allocated during initialization.

## Live Hardware Benchmark (ESP32-P4 @ 360 MHz)

Validated on the Waveshare ESP32-P4-WIFI6 development board:

| Test Case | Audio Duration | Target Class | Detection Confidence | Decision | Mean Hop Latency | Window Peak Latency |
|---|---|---|---|---|---:|---:|
| **Wake Word (`test_wake_espira.wav`)** | 2.08 s | "Hey Espira" | **100.0%** (Frame 64) | **WAKE DETECTED** | 0.35 ms | 11.83 ms |
| **Command Word (`test_cmd_licht_an.wav`)** | 1.12 s | "Licht an" | **99.4%** (Frame 63) | **CMD_1 DETECTED** | 0.32 ms | 11.77 ms |
| **Impostor / Noise (`test_background.wav`)** | 3.20 s | Negatives / Silence | Max 50.3% (< 80% Thresh) | **REJECTED (0 false alarms)** | 0.33 ms | 11.60 ms |

### Resource Consumption

- **Working State:** ~47 KB (fits comfortably inside internal SRAM, with PSRAM fallback).
- **Wake-Word Model Size:** ~58 KB (`wake_ghost.espm`, 2 classes, FP32 folded weights).
- **Command Model Size:** ~58 KB (`cmd_smarthome.espm`, 4 classes, FP32 folded weights).

## Adding to an ESP-IDF Project

Declare `esp-idf-kws-engine` and its dependencies in your application's `main/idf_component.yml`:

```yaml
dependencies:
  esp-idf-kws-engine:
    git: "https://github.com/Strg-Alt-Entf-0x00/esp-idf-kws-engine.git"
  esp-idf-model-manager:
    git: "https://github.com/Strg-Alt-Entf-0x00/esp-idf-model-manager.git"
  espressif/esp-dsp:
    version: "^1.8.2"
```

In your application `CMakeLists.txt`:

```cmake
idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES esp-idf-kws-engine
)
```

## Basic Usage

Include `esp_kws.h`, initialize the engine handle, load optional ESPM models, and stream 160-sample PCM frames:

```c
#include "esp_kws.h"
#include "esp_log.h"

static void on_kws_event(const esp_kws_result_t *res, void *user_data) {
    if (res->wake_detected) {
        ESP_LOGI("KWS", "Wake word detected! Confidence: %.1f%%", res->confidence * 100.0f);
    } else if (res->command_detected) {
        ESP_LOGI("KWS", "Command detected: '%s' (Confidence: %.1f%%)", res->label, res->confidence * 100.0f);
    }
}

void app_main(void) {
    esp_kws_config_t cfg = esp_kws_default_config();
    cfg.wake_threshold = 0.80f;
    cfg.cmd_threshold = 0.75f;
    cfg.cmd_timeout_ms = 4000;

    esp_kws_handle_t kws = NULL;
    ESP_ERROR_CHECK(esp_kws_init(&cfg, &kws));
    ESP_ERROR_CHECK(esp_kws_register_callback(kws, on_kws_event, NULL));

    // Load ESPM model files from mounted filesystem (e.g. MicroSD or SPIFFS)
    ESP_ERROR_CHECK(esp_kws_load_wake_model_file(kws, "/sd/models/wake_ghost.espm"));
    ESP_ERROR_CHECK(esp_kws_load_cmd_model_file(kws, "/sd/models/cmd_smarthome.espm"));

    // Feed 160-sample (10 ms @ 16 kHz) mono PCM frames in your audio loop
    int16_t frame[160];
    while (1) {
        // capture_microphone_pcm16(frame, 160);
        esp_kws_feed_audio(kws, frame, 160, NULL);
    }
}
```

## Python Toolchain (`tools-py/`)

The repository includes a self-contained Python toolchain under `tools-py/`:

- `model.py` — PyTorch TC-ResNet neural architecture with automatic BatchNorm folding.
- `frontend.py` — Bit-exact Python reference implementation of the 40-band Log-Mel filterbank.
- `synth_voxcpm.py` — VoxCPM2 voice cloning and multi-speaker impostor synthesizer.
- `train.py` — KWS training pipeline with acoustic data augmentation.
- `export_espm.py` — Canonical ESPM v3 container exporter with CRC32 integrity check.
- `verify_inference.py` — Bit-for-bit inference verification (Python reference vs ESPM weights).
- `build_and_train_models.py` — Autonomous end-to-end dataset synthesis, training, and model export script.

## Board Example

For a complete working application on the **Waveshare ESP32-P4-WIFI6** board featuring dual microphone support, MicroSD file management, Wi-Fi transfer, and an interactive console, refer to:

**[Waveshare ESP32-P4-WIFI6 KWS Console Example](examples/ws-esp32-p4-wifi6-kws-console-dev/README.md)**

## License

This project is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).
- **Free for noncommercial use**: Personal, educational, research, and hobbyist projects are completely free.
- **Commercial use**: Not permitted without prior authorization. For commercial inquiries, please contact the author via GitHub.
