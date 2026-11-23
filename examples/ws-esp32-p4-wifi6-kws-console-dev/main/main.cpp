/**
 * @file main.cpp
 * @brief Waveshare ESP32-P4-WIFI6 KWS & Voice Command Console Application.
 */

#include <cstdio>
#include <cstring>
#include "esp_log.h"
#include "esp_console.h"
#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_vfs_fat.h"
#include "driver/sdmmc_host.h"
#include "sdmmc_cmd.h"
#include "sd_pwr_ctrl_by_on_chip_ldo.h"
#include "nvs_flash.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "app_config.h"
#include "audio_manager.h"
#include "cmd_kws_cli.h"
#include "esp_kws.h"
#include "esp_pulsevad.h"

extern const uint8_t _binary_wake_ghost_espm_start[] asm("_binary_wake_ghost_espm_start");
extern const uint8_t _binary_wake_ghost_espm_end[]   asm("_binary_wake_ghost_espm_end");

extern const uint8_t _binary_cmd_smarthome_espm_start[] asm("_binary_cmd_smarthome_espm_start");
extern const uint8_t _binary_cmd_smarthome_espm_end[]   asm("_binary_cmd_smarthome_espm_end");

static const char *TAG = "main";

esp_kws_handle_t g_kws_handle = nullptr;
pulsevad_handle_t g_vad_handle = nullptr;
volatile bool g_live_listen_enabled = true;

// ============================================================================
// SD Card Mount for Waveshare ESP32-P4
// ============================================================================

static esp_err_t mount_sd_card(void) {
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.max_freq_khz = SDMMC_FREQ_HIGHSPEED;
    host.slot = SDMMC_HOST_SLOT_0;

    sd_pwr_ctrl_ldo_config_t ldo_config = {.ldo_chan_id = 4};
    sd_pwr_ctrl_handle_t pwr_ctrl = nullptr;
    esp_err_t err = sd_pwr_ctrl_new_on_chip_ldo(&ldo_config, &pwr_ctrl);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "On-chip LDO init for SD slot returned: %s", esp_err_to_name(err));
    } else {
        host.pwr_ctrl_handle = pwr_ctrl;
    }

    sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();
    slot_config.width = 4;

    esp_vfs_fat_sdmmc_mount_config_t mount_config = {};
    mount_config.format_if_mount_failed = false;
    mount_config.max_files = 8;
    mount_config.allocation_unit_size = 16 * 1024;
    mount_config.disk_status_check_enable = false;

    sdmmc_card_t *card = nullptr;
    err = esp_vfs_fat_sdmmc_mount(FS_MOUNT_POINT_SD, &host, &slot_config, &mount_config, &card);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "MicroSD card mounted successfully at '%s'", FS_MOUNT_POINT_SD);
        sdmmc_card_print_info(stdout, card);
    } else {
        ESP_LOGW(TAG, "MicroSD card not detected (%s).", esp_err_to_name(err));
    }
    return err;
}

// ============================================================================
// KWS Event Callback
// ============================================================================

static void on_kws_event(const esp_kws_result_t *result, void *user_data) {
    if (result->wake_detected) {
        printf("\n\033[1;32m>>> [WAKE DETECTED!] Confidence: %.1f%% | Latency: %.2f ms <<<\033[0m\n",
               result->confidence * 100.0f, result->inference_latency_ms);
        printf("\033[1;33m>>> Listening for command... <<<\033[0m\n");
    } else if (result->command_detected) {
        printf("\n\033[1;36m>>> [COMMAND DETECTED!] '%s' | Confidence: %.1f%% | Latency: %.2f ms <<<\033[0m\n",
               result->label, result->confidence * 100.0f, result->inference_latency_ms);
    }
}

// ============================================================================
// Live Audio Capture & Inference Task
// ============================================================================

static void audio_inference_task(void *pvParameters) {
    int16_t chunk[160]; // 160 samples = 10 ms @ 16 kHz

    while (1) {
        if (!g_live_listen_enabled) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        int samples_read = audio_manager_read(chunk, 160, 100);

        if (samples_read == 160) {
            if (g_kws_handle) {
                // Feed 16 kHz 160-sample PCM chunk into KWS streaming pipeline
                esp_kws_feed_audio(g_kws_handle, chunk, 160, nullptr);
            }
            vTaskDelay(pdMS_TO_TICKS(1)); // Yield to prevent CPU starvation
        } else {
            vTaskDelay(pdMS_TO_TICKS(5));
        }
    }
}

// ============================================================================
// Main Application Entry Point
// ============================================================================

extern "C" void app_main(void) {
    printf("\n");
    printf("===============================================================\n");
    printf("  Waveshare ESP32-P4-WIFI6 (Rev 1.3) KWS Console Application   \n");
    printf("  Streaming TC-ResNet Real-Time Inference (100%% Native C)      \n");
    printf("===============================================================\n\n");

    // 1. Initialize NVS (required for audio settings & Wi-Fi)
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }

    // 2. Initialize Audio Manager (ES8311 Codec + INMP441 Microphone)
    ret = audio_manager_init();
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Audio hardware initialized successfully");
        // Play acoustic feedback boot chime on speaker
        play_boot_chime();
    } else {
        ESP_LOGW(TAG, "Audio Manager init failed: %s", esp_err_to_name(ret));
    }

    // 3. Mount MicroSD card
    mount_sd_card();

    // 4. Initialize KWS Engine
    esp_kws_config_t kws_cfg = esp_kws_default_config();
    kws_cfg.wake_threshold = 0.80f;
    kws_cfg.cmd_threshold = 0.75f;
    kws_cfg.cmd_timeout_ms = 4000;
    kws_cfg.continuous_streaming = false;

    ret = esp_kws_init(&kws_cfg, &g_kws_handle);
    if (ret == ESP_OK) {
        esp_kws_register_callback(g_kws_handle, on_kws_event, nullptr);

        // Attempt loading default models from SD card first, fallback to embedded flash models
        esp_err_t load_wake = esp_kws_load_wake_model_file(g_kws_handle, "/sd/models/wake_ghost.espm");
        if (load_wake != ESP_OK) {
            size_t wake_sz = _binary_wake_ghost_espm_end - _binary_wake_ghost_espm_start;
            ESP_LOGI(TAG, "Loading embedded default wake model (%zu bytes)...", wake_sz);
            esp_kws_load_wake_model_buffer(g_kws_handle, _binary_wake_ghost_espm_start, wake_sz);
        }

        esp_err_t load_cmd = esp_kws_load_cmd_model_file(g_kws_handle, "/sd/models/cmd_smarthome.espm");
        if (load_cmd != ESP_OK) {
            size_t cmd_sz = _binary_cmd_smarthome_espm_end - _binary_cmd_smarthome_espm_start;
            ESP_LOGI(TAG, "Loading embedded default command model (%zu bytes)...", cmd_sz);
            esp_kws_load_cmd_model_buffer(g_kws_handle, _binary_cmd_smarthome_espm_start, cmd_sz);
        }
    } else {
        ESP_LOGE(TAG, "Failed to initialize KWS engine: %s", esp_err_to_name(ret));
    }

    // 5. Initialize PulseVAD (optional speech gating)
    pulsevad_config_t vad_cfg = pulsevad_default_config();
    ret = pulsevad_init(&vad_cfg, &g_vad_handle);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "PulseVAD speech detector coupled successfully");
    }

    // 6. Start Audio Capture and Inference Background Task (Core 0, Prio 4)
    audio_manager_start_capture();
    xTaskCreatePinnedToCore(audio_inference_task, "kws_audio", 4096, nullptr, 4, nullptr, 0);

    // 7. Register Console Commands and launch interactive CLI
    esp_console_repl_t *repl = nullptr;
    esp_console_repl_config_t repl_config = ESP_CONSOLE_REPL_CONFIG_DEFAULT();
    repl_config.prompt = CONSOLE_PROMPT;
    repl_config.max_cmdline_length = CONSOLE_MAX_COMMAND_LINE;

    esp_console_dev_uart_config_t uart_config = ESP_CONSOLE_DEV_UART_CONFIG_DEFAULT();
    esp_console_new_repl_uart(&uart_config, &repl_config, &repl);

    register_kws_console_commands();
    esp_console_start_repl(repl);
}
