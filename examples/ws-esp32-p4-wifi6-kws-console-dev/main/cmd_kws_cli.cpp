/**
 * @file cmd_kws_cli.cpp
 * @brief Interactive CLI Commands for KWS Engine, Microphones, Audio Playback & Recording.
 */

#include "cmd_kws_cli.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>
#include <inttypes.h>

#include "esp_log.h"
#include "esp_console.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_private/esp_clk.h"
#include "argtable3/argtable3.h"

#include "app_config.h"
#include "audio_manager.h"
#include "esp_kws.h"
#include "esp_pulsevad.h"

extern esp_kws_handle_t g_kws_handle;
extern pulsevad_handle_t g_vad_handle;
extern volatile bool g_live_listen_enabled;

#if defined(CONFIG_KWS_EMBED_TEST_AUDIO)
extern const uint8_t _binary_test_wake_espira_wav_start[] asm("_binary_test_wake_espira_wav_start");
extern const uint8_t _binary_test_wake_espira_wav_end[]   asm("_binary_test_wake_espira_wav_end");

extern const uint8_t _binary_test_cmd_licht_an_wav_start[] asm("_binary_test_cmd_licht_an_wav_start");
extern const uint8_t _binary_test_cmd_licht_an_wav_end[]   asm("_binary_test_cmd_licht_an_wav_end");

extern const uint8_t _binary_test_background_wav_start[] asm("_binary_test_background_wav_start");
extern const uint8_t _binary_test_background_wav_end[]   asm("_binary_test_background_wav_end");
#endif

// ============================================================================
// Standard WAV Header Structure
// ============================================================================

typedef struct {
    char riff[4];          // "RIFF"
    uint32_t file_size;    // 36 + data_size
    char wave[4];          // "WAVE"
    char fmt[4];           // "fmt "
    uint32_t fmt_len;      // 16
    uint16_t format;       // 1 (PCM)
    uint16_t channels;     // 1 (Mono)
    uint32_t sample_rate;  // 16000
    uint32_t byte_rate;    // sample_rate * channels * 2
    uint16_t block_align;  // channels * 2
    uint16_t bits;         // 16
    char data[4];          // "data"
    uint32_t data_size;    // raw PCM byte size
} __attribute__((packed)) wav_header_t;

// ============================================================================
// Boot Chime Synthesizer
// ============================================================================

void play_boot_chime(void) {
    audio_manager_play_boot_sequence();
}

// ============================================================================
// Command: status / info
// ============================================================================

static int cmd_status(int argc, char **argv) {
    (void)argc;
    (void)argv;

    printf("\n============================================================\n");
    printf("   ESP32-P4 Streaming KWS & Command Engine Status\n");
    printf("============================================================\n");

    // Board & Clocks
    printf("[CPU] Frequency          : %u MHz\n", (unsigned)(esp_clk_cpu_freq() / 1000000));
    printf("[MEM] Internal SRAM Free : %u KB\n", (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));
    printf("[MEM] PSRAM Free         : %u KB\n", (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));

    // Audio Codec & Microphones
    float gain = 0.0f;
    audio_manager_get_mic_gain(&gain);
    printf("[AUDIO] Driver State     : %s\n", audio_manager_is_capturing() ? "CAPTURING" : "IDLE");
    printf("[AUDIO] Codec Volume     : %u %%\n", audio_manager_get_speaker_vol());
    printf("[AUDIO] Active Mic       : %s\n",
           audio_manager_get_mic() == MIC_I2S1_INMP441 ? "INMP441 (Digital I2S1)" : "ES8311 (Onboard I2S0)");
    printf("[AUDIO] Mic Gain         : %.0f dB\n", gain);

    // KWS Engine State
    if (g_kws_handle) {
        esp_kws_state_t st = esp_kws_get_state(g_kws_handle);
        const char *st_str = "UNKNOWN";
        switch (st) {
            case ESP_KWS_STATE_IDLE: st_str = "IDLE (Listening Off)"; break;
            case ESP_KWS_STATE_WAKE_LISTENING: st_str = "WAKE_LISTENING (Continuous)"; break;
            case ESP_KWS_STATE_COMMAND_LISTENING: st_str = "COMMAND_LISTENING (Window active)"; break;
            case ESP_KWS_STATE_TRIGGERED: st_str = "TRIGGERED"; break;
            default: break;
        }
        printf("[KWS] State              : %s\n", st_str);
        printf("[KWS] Live Listening     : %s\n", g_live_listen_enabled ? "ENABLED" : "PAUSED");
    } else {
        printf("[KWS] State              : NOT INITIALIZED\n");
    }

    // PulseVAD State
    if (g_vad_handle) {
        printf("[VAD] State              : INITIALIZED (Online Pre-Filter Active)\n");
    } else {
        printf("[VAD] State              : NOT INITIALIZED\n");
    }

    printf("============================================================\n\n");
    return 0;
}

// ============================================================================
// Command: ls [path]
// ============================================================================

static struct {
    struct arg_str *path;
    struct arg_end *end;
} ls_args;

static void list_directory(const char *path, int level) {
    if (level > 3) return;
    DIR *dir = opendir(path);
    if (!dir) {
        if (level == 0) printf("Failed to open directory: %s\n", path);
        return;
    }
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        char full_path[512];
        snprintf(full_path, sizeof(full_path), "%s/%s", path, ent->d_name);
        struct stat st;
        if (stat(full_path, &st) != 0) continue;
        for (int i = 0; i < level; i++) printf("  ");
        if (S_ISDIR(st.st_mode)) {
            if (strcmp(ent->d_name, ".") != 0 && strcmp(ent->d_name, "..") != 0) {
                printf("[DIR]  %s/\n", ent->d_name);
                list_directory(full_path, level + 1);
            }
        } else {
            printf("[FILE] %-28s (%ld bytes)\n", ent->d_name, (long)st.st_size);
        }
    }
    closedir(dir);
}

static int cmd_ls(int argc, char **argv) {
    arg_parse(argc, argv, (void **)&ls_args);
    const char *path = (ls_args.path->count > 0) ? ls_args.path->sval[0] : FS_MOUNT_POINT_SD;
    printf("\nDirectory listing for '%s':\n", path);
    list_directory(path, 0);
    printf("\n");
    return 0;
}

// ============================================================================
// Command: speaker_vol
// ============================================================================

static struct {
    struct arg_int *volume;
    struct arg_end *end;
} speaker_args;

static int cmd_speaker_vol(int argc, char **argv) {
    arg_parse(argc, argv, (void **)&speaker_args);
    if (speaker_args.volume->count > 0) {
        int vol = speaker_args.volume->ival[0];
        if (vol < 0) vol = 0;
        if (vol > 100) vol = 100;
        audio_manager_set_speaker_vol((uint8_t)vol);
        printf("[OK] Speaker volume set to %d %%\n", vol);
    } else {
        printf("Speaker volume: %u %%\n", audio_manager_get_speaker_vol());
    }
    return 0;
}

// ============================================================================
// Command: play_wav <file.wav>
// ============================================================================

static struct {
    struct arg_str *file;
    struct arg_end *end;
} play_wav_args;

static int do_play_wav(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("[ERROR] Cannot open WAV file: %s\n", path);
        return 1;
    }

    wav_header_t hdr;
    if (fread(&hdr, 1, sizeof(hdr), f) != sizeof(hdr)) {
        printf("[ERROR] Failed to read WAV header\n");
        fclose(f);
        return 1;
    }

    if (memcmp(hdr.riff, "RIFF", 4) != 0 || memcmp(hdr.wave, "WAVE", 4) != 0) {
        printf("[ERROR] Invalid WAV format (missing RIFF/WAVE signature)\n");
        fclose(f);
        return 1;
    }

    printf("[INFO] Playing: %s\n", path);
    printf("       Format: %" PRIu32 " Hz, %u ch, %u bit (%" PRIu32 " data bytes)\n",
           hdr.sample_rate, (unsigned)hdr.channels, (unsigned)hdr.bits, hdr.data_size);

    bool prev_live = g_live_listen_enabled;
    g_live_listen_enabled = false;

    esp_err_t ret = audio_manager_start_playback();
    if (ret != ESP_OK) {
        printf("[ERROR] Failed to start audio playback: %s\n", esp_err_to_name(ret));
        fclose(f);
        g_live_listen_enabled = prev_live;
        return 1;
    }

    int16_t buffer[512];
    size_t samples_read = 0;
    while ((samples_read = fread(buffer, sizeof(int16_t), 512, f)) > 0) {
        audio_manager_write(buffer, (int)samples_read, 1000);
    }

    vTaskDelay(pdMS_TO_TICKS(100));
    audio_manager_stop_playback();
    fclose(f);
    g_live_listen_enabled = prev_live;

    printf("[OK] Playback finished\n");
    return 0;
}

static int cmd_play_wav(int argc, char **argv) {
    int nerrors = arg_parse(argc, argv, (void **)&play_wav_args);
    if (nerrors != 0) {
        arg_print_errors(stdout, play_wav_args.end, "play_wav");
        printf("Usage: play_wav <file.wav>\n");
        return 1;
    }
    return do_play_wav(play_wav_args.file->sval[0]);
}

// ============================================================================
// Command: record_mic <output.wav> <seconds>
// ============================================================================

static struct {
    struct arg_str *file;
    struct arg_int *duration;
    struct arg_end *end;
} record_mic_args;

static int do_record_mic(const char *output_path, int duration_sec) {
    if (duration_sec <= 0 || duration_sec > 60) {
        printf("[ERROR] Recording duration must be between 1 and 60 seconds\n");
        return 1;
    }

    FILE *f = fopen(output_path, "wb");
    if (!f) {
        printf("[ERROR] Failed to open file for writing: %s\n", output_path);
        return 1;
    }

    wav_header_t hdr = {};
    memcpy(hdr.riff, "RIFF", 4);
    memcpy(hdr.wave, "WAVE", 4);
    memcpy(hdr.fmt, "fmt ", 4);
    hdr.fmt_len = 16;
    hdr.format = 1; // PCM
    hdr.channels = 1;
    hdr.sample_rate = 16000;
    hdr.bits = 16;
    hdr.byte_rate = 16000 * 1 * sizeof(int16_t);
    hdr.block_align = 1 * sizeof(int16_t);
    memcpy(hdr.data, "data", 4);
    hdr.data_size = 0;
    hdr.file_size = 36;
    fwrite(&hdr, 1, sizeof(hdr), f);

    uint32_t total_samples = duration_sec * 16000;
    uint32_t samples_recorded = 0;

    printf("[INFO] Recording %d seconds to: %s\n", duration_sec, output_path);
    printf("       Microphone: %s\n",
           audio_manager_get_mic() == MIC_I2S1_INMP441 ? "INMP441 (Digital I2S1)" : "ES8311 (Onboard I2S0)");

    bool prev_live = g_live_listen_enabled;
    g_live_listen_enabled = false;
    vTaskDelay(pdMS_TO_TICKS(15)); // Allow inference task to sleep

    audio_manager_start_capture();

    int16_t buffer[320];
    while (samples_recorded < total_samples) {
        int to_read = (int)std::min<uint32_t>(total_samples - samples_recorded, 320);
        int n = audio_manager_read(buffer, to_read, 1000);
        if (n > 0) {
            fwrite(buffer, sizeof(int16_t), n, f);
            samples_recorded += n;
            if (samples_recorded % 16000 == 0) {
                printf("\r[RECORDING] %" PRIu32 "/%d seconds...", samples_recorded / 16000, duration_sec);
                fflush(stdout);
            }
        } else {
            vTaskDelay(pdMS_TO_TICKS(2));
        }
    }
    printf("\n");

    // Finalize WAV header
    hdr.data_size = samples_recorded * sizeof(int16_t);
    hdr.file_size = hdr.data_size + 36;
    fseek(f, 0, SEEK_SET);
    fwrite(&hdr, 1, sizeof(hdr), f);
    fclose(f);

    g_live_listen_enabled = prev_live;

    printf("[OK] Recording saved: %s (%" PRIu32 " samples, %.1f sec)\n",
           output_path, samples_recorded, (float)samples_recorded / 16000.0f);
    return 0;
}

static int cmd_record_mic(int argc, char **argv) {
    int nerrors = arg_parse(argc, argv, (void **)&record_mic_args);
    if (nerrors != 0) {
        arg_print_errors(stdout, record_mic_args.end, "record_mic");
        printf("Usage: record_mic <output.wav> <seconds>\n");
        return 1;
    }
    return do_record_mic(record_mic_args.file->sval[0], record_mic_args.duration->ival[0]);
}

// ============================================================================
// Command: mic
// ============================================================================

static struct {
    struct arg_str *action;
    struct arg_str *value;
    struct arg_end *end;
} mic_args;

static int cmd_mic(int argc, char **argv) {
    int nerrors = arg_parse(argc, argv, (void **)&mic_args);
    if (nerrors != 0) {
        arg_print_errors(stdout, mic_args.end, "mic");
        printf("Usage: mic status | mic select <es8311|inmp441> | mic gain <0-42> | mic test <seconds>\n");
        return 1;
    }

    const char *action = mic_args.action->sval[0];

    if (strcmp(action, "status") == 0) {
        float gain = 0.0f;
        audio_manager_get_mic_gain(&gain);
        printf("Active Mic: %s\n",
               audio_manager_get_mic() == MIC_I2S1_INMP441 ? "INMP441" : "ES8311");
        printf("ES8311 Mic Gain: %.0f dB\n", gain);
        return 0;
    }

    if (strcmp(action, "select") == 0) {
        if (mic_args.value->count == 0) {
            printf("Error: specify 'es8311' or 'inmp441'\n");
            return 1;
        }
        const char *val = mic_args.value->sval[0];
        if (strcmp(val, "es8311") == 0) {
            audio_manager_set_mic(MIC_I2S0_ES8311);
            printf("[OK] Switched to Onboard ES8311 Microphone\n");
        } else if (strcmp(val, "inmp441") == 0) {
            audio_manager_set_mic(MIC_I2S1_INMP441);
            printf("[OK] Switched to INMP441 Digital Microphone\n");
        } else {
            printf("Invalid mic: %s\n", val);
            return 1;
        }
        return 0;
    }

    if (strcmp(action, "gain") == 0) {
        if (mic_args.value->count == 0) {
            printf("Error: specify gain dB (0, 6, 12, 18, 24, 30, 36, 42)\n");
            return 1;
        }
        float gain = (float)atof(mic_args.value->sval[0]);
        esp_err_t ret = audio_manager_set_mic_gain(gain);
        if (ret == ESP_OK) {
            printf("[OK] Set mic gain to %.0f dB\n", gain);
        } else {
            printf("[ERROR] Failed to set gain: %d (Allowed: 0, 6, 12, 18, 24, 30, 36, 42)\n", ret);
        }
        return 0;
    }

    if (strcmp(action, "test") == 0) {
        int sec = 3;
        if (mic_args.value->count > 0) {
            sec = atoi(mic_args.value->sval[0]);
            if (sec <= 0 || sec > 30) sec = 3;
        }
        printf("[MIC TEST] Sampling audio for %d seconds...\n", sec);

        bool prev_live = g_live_listen_enabled;
        g_live_listen_enabled = false;
        vTaskDelay(pdMS_TO_TICKS(15));

        int total_chunks = sec * 100; // 10ms chunks (160 samples @ 16 kHz)
        int64_t sum_sq = 0;
        int peak = 0;
        int samples_read = 0;

        for (int c = 0; c < total_chunks; c++) {
            int16_t chunk[160];
            int n = audio_manager_read(chunk, 160, 100);
            if (n > 0) {
                for (int i = 0; i < n; i++) {
                    int16_t s = chunk[i];
                    sum_sq += (int64_t)s * s;
                    if (abs(s) > peak) peak = abs(s);
                }
                samples_read += n;
            }
        }

        g_live_listen_enabled = prev_live;

        float rms = sqrtf((float)sum_sq / (float)std::max(samples_read, 1));
        float peak_db = 20.0f * log10f(((float)peak + 1.0f) / 32768.0f);
        float rms_db = 20.0f * log10f((rms + 1.0f) / 32768.0f);

        printf("[MIC TEST] Done! Peak: %d (%.1f dBFS) | RMS: %.1f (%.1f dBFS)\n",
               peak, peak_db, rms, rms_db);
        return 0;
    }

    printf("Unknown mic action: %s\n", action);
    return 1;
}

// ============================================================================
// Command: audio (Unified Audio Subcommands)
// ============================================================================

static struct {
    struct arg_str *action;
    struct arg_str *arg1;
    struct arg_str *arg2;
    struct arg_end *end;
} audio_cli_args;

static int cmd_audio(int argc, char **argv) {
    int nerrors = arg_parse(argc, argv, (void **)&audio_cli_args);
    if (nerrors != 0) {
        arg_print_errors(stdout, audio_cli_args.end, "audio");
        printf("Usage:\n");
        printf("  audio status\n");
        printf("  audio volume <0-100>\n");
        printf("  audio gain <0-42>\n");
        printf("  audio play <file.wav>\n");
        printf("  audio record <file.wav> [seconds]\n");
        return 1;
    }

    const char *action = audio_cli_args.action->sval[0];

    if (strcmp(action, "status") == 0) {
        float gain = 0.0f;
        audio_manager_get_mic_gain(&gain);
        printf("[AUDIO] Speaker Volume : %u %%\n", audio_manager_get_speaker_vol());
        printf("[AUDIO] Active Mic     : %s\n",
               audio_manager_get_mic() == MIC_I2S1_INMP441 ? "INMP441 (Digital I2S1)" : "ES8311 (Onboard I2S0)");
        printf("[AUDIO] Mic Gain       : %.0f dB\n", gain);
        printf("[AUDIO] Capture Active : %s\n", audio_manager_is_capturing() ? "YES" : "NO");
        printf("[AUDIO] Playback Active: %s\n", audio_manager_is_playing() ? "YES" : "NO");
        return 0;
    }

    if (strcmp(action, "volume") == 0) {
        if (audio_cli_args.arg1->count == 0) {
            printf("Speaker volume: %u %%\n", audio_manager_get_speaker_vol());
            return 0;
        }
        int vol = atoi(audio_cli_args.arg1->sval[0]);
        if (vol < 0) vol = 0;
        if (vol > 100) vol = 100;
        audio_manager_set_speaker_vol((uint8_t)vol);
        printf("[OK] Speaker volume set to %d %%\n", vol);
        return 0;
    }

    if (strcmp(action, "gain") == 0) {
        if (audio_cli_args.arg1->count == 0) {
            float g = 0.0f;
            audio_manager_get_mic_gain(&g);
            printf("ES8311 mic gain: %.0f dB\n", g);
            return 0;
        }
        float gain = (float)atof(audio_cli_args.arg1->sval[0]);
        esp_err_t ret = audio_manager_set_mic_gain(gain);
        if (ret == ESP_OK) {
            printf("[OK] Set mic gain to %.0f dB\n", gain);
        } else {
            printf("[ERROR] Failed to set mic gain: %d (Allowed: 0, 6, 12, 18, 24, 30, 36, 42)\n", ret);
        }
        return 0;
    }

    if (strcmp(action, "play") == 0) {
        if (audio_cli_args.arg1->count == 0) {
            printf("Usage: audio play <file.wav>\n");
            return 1;
        }
        return do_play_wav(audio_cli_args.arg1->sval[0]);
    }

    if (strcmp(action, "record") == 0) {
        if (audio_cli_args.arg1->count == 0) {
            printf("Usage: audio record <file.wav> [seconds]\n");
            return 1;
        }
        int sec = 3;
        if (audio_cli_args.arg2->count > 0) {
            sec = atoi(audio_cli_args.arg2->sval[0]);
        }
        return do_record_mic(audio_cli_args.arg1->sval[0], sec);
    }

    printf("Unknown audio action: %s\n", action);
    return 1;
}

// ============================================================================
// Command: kws (Keyword Spotting Engine Controls)
// ============================================================================

static void run_wav_buffer_test(const char *name, const uint8_t *wav_data, size_t size, bool start_in_cmd_mode = false) {
    if (!wav_data || size <= 44) return;
    const int16_t *samples = (const int16_t *)(wav_data + 44);
    size_t num_samples = (size - 44) / sizeof(int16_t);
    size_t num_frames = num_samples / 160;

    printf("\n>>> Benchmark Test: '%s' (%zu frames = %.2f seconds) <<<\n",
           name, num_frames, (float)num_samples / 16000.0f);

    bool prev_live = g_live_listen_enabled;
    g_live_listen_enabled = false;
    vTaskDelay(pdMS_TO_TICKS(15));

    esp_kws_reset(g_kws_handle);
    if (start_in_cmd_mode) {
        esp_kws_start_command_listening(g_kws_handle);
    } else {
        esp_kws_start_listening(g_kws_handle);
    }

    int wakes = 0;
    int cmds = 0;
    int64_t total_us = 0;
    float peak_target_conf = 0.0f;
    size_t peak_target_frame = 0;
    uint16_t peak_target_class = 0;

    for (size_t f = 0; f < num_frames; f++) {
        esp_kws_result_t res = {};
        int64_t t0 = esp_timer_get_time();
        esp_kws_feed_audio(g_kws_handle, samples + f * 160, 160, &res);
        int64_t elapsed = esp_timer_get_time() - t0;
        total_us += elapsed;

        float target_prob = (res.num_classes > 1) ? res.all_probs[1] : 0.0f;
        if (target_prob > peak_target_conf) {
            peak_target_conf = target_prob;
            peak_target_frame = f;
            peak_target_class = 1;
        }

        if (target_prob >= 0.20f) {
            printf("  [Frame %3zu | %5.2fs] Target Prob: %5.1f%% (Bg: %5.1f%%, Latency: %.2f ms)\n",
                   f, (float)f * 0.01f, target_prob * 100.0f, res.all_probs[0] * 100.0f, res.inference_latency_ms);
        }

        if (res.wake_detected) {
            printf("  [Frame %3zu | %5.2fs] >>> WAKE DETECTED! (Conf: %5.1f%%, Latency: %.2f ms) <<<\n",
                   f, (float)f * 0.01f, res.confidence * 100.0f, res.inference_latency_ms);
            wakes++;
        }
        if (res.command_detected) {
            printf("  [Frame %3zu | %5.2fs] >>> COMMAND DETECTED: '%s' (Conf: %5.1f%%, Latency: %.2f ms) <<<\n",
                   f, (float)f * 0.01f, res.label, res.confidence * 100.0f, res.inference_latency_ms);
            cmds++;
        }
    }

    g_live_listen_enabled = prev_live;

    float avg_ms = num_frames > 0 ? (float)total_us / (float)num_frames / 1000.0f : 0.0f;
    printf("Result: %d Wakes, %d Commands | Target Peak: Class #%u (%.1f%% @ %.2fs) | Avg: %.2f ms / frame (Budget: 10.0 ms)\n",
           wakes, cmds, peak_target_class, peak_target_conf * 100.0f, (float)peak_target_frame * 0.01f, avg_ms);
}

static struct {
    struct arg_str *cmd;
    struct arg_str *arg;
    struct arg_end *end;
} kws_cli_args;

static int cmd_kws(int argc, char **argv) {
    int nerrors = arg_parse(argc, argv, (void **)&kws_cli_args);
    if (nerrors != 0) {
        arg_print_errors(stdout, kws_cli_args.end, "kws");
        printf("Usage:\n");
        printf("  kws status\n");
        printf("  kws listen [on|off]\n");
        printf("  kws benchmark\n");
        printf("  kws load_wake <file.espm>\n");
        printf("  kws load_cmd  <file.espm>\n");
        printf("  kws test_wav  <file.wav>\n");
        return 1;
    }

    const char *cmd = kws_cli_args.cmd->sval[0];

    if (strcmp(cmd, "status") == 0) {
        cmd_status(0, NULL);
        return 0;
    }

    if (strcmp(cmd, "benchmark") == 0) {
        printf("============================================================\n");
        printf("   ESP32-P4 Embedded KWS Engine Latency & Accuracy Benchmark\n");
        printf("============================================================\n");
#if defined(CONFIG_KWS_EMBED_TEST_AUDIO)
        size_t sz_wake = _binary_test_wake_espira_wav_end - _binary_test_wake_espira_wav_start;
        run_wav_buffer_test("test_wake_espira.wav (Target: Hey Espira)", _binary_test_wake_espira_wav_start, sz_wake, false);

        size_t sz_cmd = _binary_test_cmd_licht_an_wav_end - _binary_test_cmd_licht_an_wav_start;
        run_wav_buffer_test("test_cmd_licht_an.wav (Target: Licht an)", _binary_test_cmd_licht_an_wav_start, sz_cmd, true);

        size_t sz_bg = _binary_test_background_wav_end - _binary_test_background_wav_start;
        run_wav_buffer_test("test_background.wav (Impostor/Noise Rejection)", _binary_test_background_wav_start, sz_bg, false);
#else
        printf("[INFO] Test audio files not embedded into flash.\n");
        printf("You can test any WAV file from SD card using:\n");
        printf("  kws test_wav /sd/audio/<filename.wav>\n");
#endif
        printf("\n============================================================\n\n");
        return 0;
    }

    if (strcmp(cmd, "listen") == 0) {
        if (kws_cli_args.arg->count > 0 && strcmp(kws_cli_args.arg->sval[0], "off") == 0) {
            g_live_listen_enabled = false;
            esp_kws_stop_listening(g_kws_handle);
            printf("[OK] Live KWS listening PAUSED\n");
        } else {
            g_live_listen_enabled = true;
            esp_kws_start_listening(g_kws_handle);
            printf("[OK] Live KWS listening ACTIVE\n");
        }
        return 0;
    }

    if (strcmp(cmd, "load_wake") == 0) {
        if (kws_cli_args.arg->count == 0) {
            printf("Usage: kws load_wake <file.espm>\n");
            return 1;
        }
        const char *p = kws_cli_args.arg->sval[0];
        esp_err_t ret = esp_kws_load_wake_model_file(g_kws_handle, p);
        if (ret == ESP_OK) {
            printf("[OK] Loaded wake model: '%s'\n", p);
        } else {
            printf("[ERROR] Failed to load wake model '%s': %d\n", p, ret);
        }
        return 0;
    }

    if (strcmp(cmd, "load_cmd") == 0) {
        if (kws_cli_args.arg->count == 0) {
            printf("Usage: kws load_cmd <file.espm>\n");
            return 1;
        }
        const char *p = kws_cli_args.arg->sval[0];
        esp_err_t ret = esp_kws_load_cmd_model_file(g_kws_handle, p);
        if (ret == ESP_OK) {
            printf("[OK] Loaded command model: '%s'\n", p);
        } else {
            printf("[ERROR] Failed to load command model '%s': %d\n", p, ret);
        }
        return 0;
    }

    if (strcmp(cmd, "test_wav") == 0) {
        if (kws_cli_args.arg->count == 0) {
            printf("Usage: kws test_wav <file.wav>\n");
            return 1;
        }
        const char *path = kws_cli_args.arg->sval[0];
        FILE *f = fopen(path, "rb");
        if (!f) {
            printf("[ERROR] Cannot open file: %s\n", path);
            return 1;
        }

        fseek(f, 44, SEEK_SET);
        int16_t chunk[160];
        int frame_idx = 0;
        int wakes = 0;
        int cmds = 0;
        int64_t total_inference_us = 0;

        printf("[WAV TEST] Streaming '%s' through KWS Engine...\n", path);
        bool prev_live = g_live_listen_enabled;
        g_live_listen_enabled = false;
        vTaskDelay(pdMS_TO_TICKS(15));

        esp_kws_reset(g_kws_handle);
        esp_kws_start_listening(g_kws_handle);

        while (fread(chunk, sizeof(int16_t), 160, f) == 160) {
            esp_kws_result_t res = {};
            int64_t t0 = esp_timer_get_time();
            esp_kws_feed_audio(g_kws_handle, chunk, 160, &res);
            int64_t elapsed = esp_timer_get_time() - t0;
            total_inference_us += elapsed;

            if (res.wake_detected) {
                printf("  [Frame %d | %.2fs] WAKE DETECTED! (Conf: %.1f%%, Latency: %.2f ms)\n",
                       frame_idx, (float)frame_idx * 0.01f, res.confidence * 100.0f, res.inference_latency_ms);
                wakes++;
            }
            if (res.command_detected) {
                printf("  [Frame %d | %.2fs] COMMAND DETECTED: '%s' (Conf: %.1f%%, Latency: %.2f ms)\n",
                       frame_idx, (float)frame_idx * 0.01f, res.label, res.confidence * 100.0f, res.inference_latency_ms);
                cmds++;
            }
            frame_idx++;
        }
        fclose(f);
        g_live_listen_enabled = prev_live;

        float avg_lat = frame_idx > 0 ? (float)total_inference_us / (float)frame_idx / 1000.0f : 0.0f;
        printf("[WAV TEST] Benchmark finished!\n");
        printf("  Processed Frames      : %d (%.2f seconds of audio)\n", frame_idx, (float)frame_idx * 0.01f);
        printf("  Avg Inference Latency : %.2f ms / frame (Real-time budget: 10.0 ms)\n", avg_lat);
        printf("  Total Detections      : %d Wakes, %d Commands\n", wakes, cmds);
        return 0;
    }

    printf("Unknown kws command: %s\n", cmd);
    return 1;
}

// ============================================================================
// Command Registration
// ============================================================================

esp_err_t register_kws_console_commands(void) {
    // 1. status / info
    esp_console_cmd_t status_cmd = {};
    status_cmd.command = "status";
    status_cmd.help = "Print ESP32-P4 hardware and audio status";
    status_cmd.func = &cmd_status;
    esp_console_cmd_register(&status_cmd);

    esp_console_cmd_t info_cmd = {};
    info_cmd.command = "info";
    info_cmd.help = "Print system information (alias of status)";
    info_cmd.func = &cmd_status;
    esp_console_cmd_register(&info_cmd);

    // 2. ls
    ls_args.path = arg_str0(NULL, NULL, "<path>", "Directory path (default: /sd)");
    ls_args.end = arg_end(1);
    esp_console_cmd_t ls_cmd = {};
    ls_cmd.command = "ls";
    ls_cmd.help = "List directory contents on SD card or VFS";
    ls_cmd.func = &cmd_ls;
    ls_cmd.argtable = &ls_args;
    esp_console_cmd_register(&ls_cmd);

    // 3. mic
    mic_args.action = arg_str1(NULL, NULL, "<action>", "status | select | gain | test");
    mic_args.value = arg_str0(NULL, NULL, "<val>", "value");
    mic_args.end = arg_end(2);
    esp_console_cmd_t mic_cmd = {};
    mic_cmd.command = "mic";
    mic_cmd.help = "Microphone configuration and capture diagnostics";
    mic_cmd.func = &cmd_mic;
    mic_cmd.argtable = &mic_args;
    esp_console_cmd_register(&mic_cmd);

    // 4. speaker_vol
    speaker_args.volume = arg_int0(NULL, NULL, "<vol>", "Volume 0-100 (default: query)");
    speaker_args.end = arg_end(1);
    esp_console_cmd_t spk_cmd = {};
    spk_cmd.command = "speaker_vol";
    spk_cmd.help = "Get or set ES8311 speaker volume (0-100%)";
    spk_cmd.func = &cmd_speaker_vol;
    spk_cmd.argtable = &speaker_args;
    esp_console_cmd_register(&spk_cmd);

    // 5. play_wav
    play_wav_args.file = arg_str1(NULL, NULL, "<file.wav>", "WAV file path");
    play_wav_args.end = arg_end(1);
    esp_console_cmd_t play_cmd = {};
    play_cmd.command = "play_wav";
    play_cmd.help = "Play 16 kHz mono WAV file through ES8311 speaker";
    play_cmd.func = &cmd_play_wav;
    play_cmd.argtable = &play_wav_args;
    esp_console_cmd_register(&play_cmd);

    // 6. record_mic
    record_mic_args.file = arg_str1(NULL, NULL, "<output.wav>", "Output WAV file path");
    record_mic_args.duration = arg_int1(NULL, NULL, "<seconds>", "Recording duration in seconds (1-60)");
    record_mic_args.end = arg_end(2);
    esp_console_cmd_t rec_cmd = {};
    rec_cmd.command = "record_mic";
    rec_cmd.help = "Record audio from active microphone to WAV file";
    rec_cmd.func = &cmd_record_mic;
    rec_cmd.argtable = &record_mic_args;
    esp_console_cmd_register(&rec_cmd);

    // 7. audio (unified command group)
    audio_cli_args.action = arg_str1(NULL, NULL, "<status|volume|gain|play|record>", "Action");
    audio_cli_args.arg1 = arg_str0(NULL, NULL, "<arg1>", "File path or value");
    audio_cli_args.arg2 = arg_str0(NULL, NULL, "<arg2>", "Duration or extra value");
    audio_cli_args.end = arg_end(3);
    esp_console_cmd_t audio_cmd = {};
    audio_cmd.command = "audio";
    audio_cmd.help = "Unified audio control (status, volume, gain, play, record)";
    audio_cmd.func = &cmd_audio;
    audio_cmd.argtable = &audio_cli_args;
    esp_console_cmd_register(&audio_cmd);

    // 8. kws
    kws_cli_args.cmd = arg_str1(NULL, NULL, "<status|benchmark|listen|load_wake|load_cmd|test_wav>", "KWS subcommand");
    kws_cli_args.arg = arg_str0(NULL, NULL, "<arg>", "Argument");
    kws_cli_args.end = arg_end(2);
    esp_console_cmd_t kws_cmd = {};
    kws_cmd.command = "kws";
    kws_cmd.help = "Keyword Spotting engine controls, benchmark and WAV tester";
    kws_cmd.func = &cmd_kws;
    kws_cmd.argtable = &kws_cli_args;
    esp_console_cmd_register(&kws_cmd);

    return ESP_OK;
}
