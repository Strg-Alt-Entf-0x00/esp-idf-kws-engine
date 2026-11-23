/**
 * @file esp_kws_types.h
 * @brief Type definitions and constants for the ESP-IDF KWS Engine.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ESP_KWS_SAMPLE_RATE_HZ         16000
#define ESP_KWS_FRAME_HOP_SAMPLES      160     /* 10 ms hop */
#define ESP_KWS_WIN_LENGTH_SAMPLES     480     /* 30 ms window */
#define ESP_KWS_N_FFT                  512
#define ESP_KWS_NUM_MEL_BINS           40
#define KWS_NUM_MEL_BINS               40
#define ESP_KWS_MAX_LABEL_LEN          32
#define ESP_KWS_MAX_CLASSES            32

/**
 * @brief Engine operational states.
 */
typedef enum {
    ESP_KWS_STATE_UNINITIALIZED = 0,
    ESP_KWS_STATE_IDLE,
    ESP_KWS_STATE_WAKE_LISTENING,
    ESP_KWS_STATE_COMMAND_LISTENING,
    ESP_KWS_STATE_TRIGGERED
} esp_kws_state_t;

/**
 * @brief Detection event result payload.
 */
typedef struct {
    bool wake_detected;
    bool command_detected;
    char label[ESP_KWS_MAX_LABEL_LEN];
    uint16_t class_index;
    float confidence;
    float inference_latency_ms;
    uint16_t num_classes;
    float all_probs[ESP_KWS_MAX_CLASSES];
} esp_kws_result_t;

/**
 * @brief Configuration parameters for initializing a KWS instance.
 */
typedef struct {
    float wake_threshold;         /**< Confidence threshold (0.0 to 1.0) */
    float cmd_threshold;          /**< Confidence threshold (0.0 to 1.0) */
    uint32_t cmd_timeout_ms;      /**< Timeout in ms for commands after wake */
    bool continuous_streaming;    /**< True if frames stream continuously without VAD gating */
} esp_kws_config_t;

/**
 * @brief Returns the default recommended configuration.
 */
static inline esp_kws_config_t esp_kws_default_config(void) {
    esp_kws_config_t cfg = {
        .wake_threshold = 0.80f,
        .cmd_threshold = 0.75f,
        .cmd_timeout_ms = 3500,
        .continuous_streaming = false
    };
    return cfg;
}

#ifdef __cplusplus
}
#endif
