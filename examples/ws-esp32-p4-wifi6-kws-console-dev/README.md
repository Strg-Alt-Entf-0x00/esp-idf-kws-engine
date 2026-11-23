# Waveshare ESP32-P4-WIFI6 KWS Console Example

This example demonstrates the `esp-idf-kws-engine` running on the **Waveshare ESP32-P4-WIFI6** (Rev 1.3) development board. It includes dual-microphone audio capture, acoustic boot chime playback, MicroSD storage, and an interactive serial console for real-time keyword spotting and voice command evaluation.

Board-specific hardware setup and console commands are documented here; the [root README](../../README.md) describes the core `esp-idf-kws-engine` component and its public C API.

## Requirements

- **Development Board:** Waveshare ESP32-P4-WIFI6 (Rev 1.3 @ 360 MHz, 32 MB PSRAM @ 200 MHz)
- **Framework:** ESP-IDF v6.0.2 or later
- **MicroSD Card:** Formatted as FAT32 for models and audio recordings
- **Serial Connection:** UART0 connection for the interactive console (115200 baud)
- **External Microphone (Optional):** INMP441 I2S MEMS microphone for external audio testing

## Build and Flash

From this example's project directory:

```bash
idf.py build
idf.py -p PORT flash
idf.py -p PORT monitor
```

Replace `PORT` with the serial port of your board (e.g. `COM4` on Windows or `/dev/ttyUSB0` on Linux).

On Windows, the optional [`esp-build.bat`](esp-build.bat) script provides a standalone build and flash wrapper:

```cmd
esp-build.bat build
esp-build.bat flash
esp-build.bat monitor
```

## Storage and Model Files

The application automatically mounts the MicroSD card at `/sd` on boot using the ESP32-P4 on-chip LDO power controller (LDO Channel 4). It utilizes the following directory structure:

- `/sd/models/` — ESPM keyword spotting and command models (`*.espm`)
- `/sd/audio/` — WAV recordings and test audio playback files

### Pre-trained Models

Canonical pre-trained ESPM models are provided in the [`models/espm/`](../../models/espm/) directory at repository root:

- `wake_ghost.espm` — High-accuracy wake word model ("Hey Espira" / "Ghost")
- `cmd_smarthome.espm` — Multi-class smart home command model ("Licht an", "Licht aus", "Stop")

To load these models dynamically from SD card, copy the `.espm` files into `/sd/models/` on your MicroSD card.

> [!NOTE]
> **Embedded Fallback:** If no SD card is detected or the model files are not found in `/sd/models/`, the application automatically falls back to the built-in models embedded in flash memory.

## Audio Hardware & Acoustic Feedback

1. **Boot Chime:** Upon successful initialization of the ES8311 audio codec and PA amplifier (GPIO 53), the onboard speaker plays a polyphonic acoustic boot chime signaling that the system is ready for audio capture.
2. **Dual Microphone Architecture:**
   - **Onboard ES8311 Codec Microphone (I2S0):** Built-in analog electret microphone with configurable analog PGA gain (0–42 dB).
   - **External INMP441 MEMS Microphone (I2S1):** Digital I2S microphone for high-SNR directional capture.
   - Microphones can be toggled at runtime using the `mic select` command. Settings are saved to NVS and persist across reboots.

## Interactive Console Commands

Open the serial monitor at 115200 baud to interact with the `kws_dev>` shell:

| Command | Arguments | Description |
|---|---|---|
| `status` / `info` | - | Prints hardware status, CPU frequency, free SRAM/PSRAM, and audio state |
| `ls` | `[path]` | Lists files and directories on the MicroSD card (default: `/sd`) |
| `speaker_vol` | `[0-100]` | Queries or sets the ES8311 speaker volume in percent (default: 50%) |
| `play_wav` | `<path.wav>` | Plays a 16 kHz mono WAV file from SD card through the speaker |
| `record_mic` | `<output.wav> <seconds>` | Records raw 16 kHz audio from the active microphone directly to SD card |
| `mic status` | - | Reports the active microphone, channel, and analog gain |
| `mic select` | `onboard` \| `i2s` | Switches active input between onboard ES8311 and external INMP441 |
| `mic gain` | `0`..`42` | Sets the analog preamp gain of the ES8311 codec (in 6 dB steps) |
| `mic test` | `[seconds]` | Runs live audio level diagnostics (Peak dBFS and RMS meters) |
| `kws status` | - | Displays engine state, detection thresholds, and loaded models |
| `kws listen` | `on` \| `off` | Starts or pauses continuous live microphone inference |
| `kws benchmark` | - | Runs automated real-time latency and accuracy benchmark across test audio |
| `kws test_wav` | `<file.wav>` | Runs offline inference on a WAV file from the SD card |
| `kws wake_thresh` | `<0.10-0.99>` | Adjusts sensitivity threshold for the wake word model |
| `kws cmd_thresh` | `<0.10-0.99>` | Adjusts sensitivity threshold for command word models |
| `kws listen_cmd` | - | Forces the engine into command listening mode (5-second window) |
| `kws load_wake` | `<path.espm>` | Dynamically loads a new wake word model from SD card |
| `kws load_cmd` | `<path.espm>` | Dynamically loads a new command word model from SD card |

## Hardware Interfaces & Pinout

The following board pins are configured in `board_audio_config.h` and `sdkconfig.defaults`:

| Interface | Peripheral | Signal / Pin | Details |
|---|---|---|---|
| **MicroSD Slot** | SDMMC Host Slot 0 | CLK (43), CMD (44), D0–D3 (39–42) | 4-bit bus, powered by P4 on-chip LDO channel 4 |
| **ES8311 Audio Codec** | I2C0 & I2S0 | SDA (8), SCL (7), MCLK (13), BCLK (12), WS (10), DOUT (9), DIN (11) | Master mode, 16 kHz 16-bit PCM |
| **Speaker PA Enable** | GPIO | GPIO 53 | High = Amplifier enabled |
| **INMP441 Microphone** | I2S1 | BCLK (20), WS (21), DIN (22) | 16 kHz, 32-bit Philips format, 16-bit high data |
| **Wi-Fi 6 Co-processor** | ESP-Hosted | SDIO to onboard ESP32-C6 | Configured via ESP-Hosted slave link |
| **Serial Console** | UART0 | TX (35), RX (36) | 115200 baud, 8-N-1 |

## Performance Benchmark

Measured on the Waveshare ESP32-P4-WIFI6 (Rev 1.3 @ 360 MHz):

```text
kws_dev> kws benchmark
============================================================
           1D TC-ResNet Streaming KWS Benchmark             
============================================================
Model: wake_ghost (Classes: 2, Weights: 58.2 KB, Context: 64 frames)
Hop size: 160 samples (10.0 ms @ 16 kHz)
------------------------------------------------------------
Audio: test_wake_espira.wav (2.08 s, 33280 samples, 208 frames)
  Mean Hop Latency:   0.35 ms
  Peak Window Latency: 11.83 ms
  Result:             WAKE DETECTED (Frame 64, Conf: 100.0%)
------------------------------------------------------------
Audio: test_background.wav (3.20 s, 51200 samples, 320 frames)
  Mean Hop Latency:   0.33 ms
  Result:             REJECTED / NO ALARM (Max Conf: 50.3%)
============================================================
```
