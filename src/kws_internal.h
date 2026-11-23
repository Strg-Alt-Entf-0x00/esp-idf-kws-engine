/**
 * @file kws_internal.h
 * @brief Internal Data Structures for KWS Engine.
 */

#pragma once

#include "esp_kws.h"
#include "kws_frontend.h"
#include "kws_nn.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

struct esp_kws_engine_s {
    esp_kws_config_t config;
    esp_kws_state_t state;

    kws_frontend_t frontend;
    kws_nn_state_t *nn_state;

    kws_nn_weights_t wake_weights;
    kws_nn_weights_t cmd_weights;
    bool wake_model_loaded;
    bool cmd_model_loaded;

    int16_t hop_buffer[ESP_KWS_FRAME_HOP_SAMPLES];
    size_t hop_buf_len;

    esp_kws_callback_t callback;
    void *callback_ctx;

    int64_t cmd_start_time_us;
    SemaphoreHandle_t mutex;
};
