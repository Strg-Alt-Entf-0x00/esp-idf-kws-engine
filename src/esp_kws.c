/**
 * @file esp_kws.c
 * @brief Core State Machine and Audio Orchestrator for ESP-IDF KWS Engine.
 */

#include "kws_internal.h"
#include "loaders/kws_espm_loader.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <string.h>
#include <stdlib.h>

static const char *TAG = "ESP_KWS";

esp_err_t esp_kws_init(const esp_kws_config_t *config, esp_kws_handle_t *out_handle) {
    if (!out_handle) {
        return ESP_ERR_INVALID_ARG;
    }

    struct esp_kws_engine_s *eng = (struct esp_kws_engine_s *)heap_caps_calloc(
        1, sizeof(struct esp_kws_engine_s), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL
    );
    if (!eng) {
        eng = (struct esp_kws_engine_s *)heap_caps_calloc(1, sizeof(struct esp_kws_engine_s), MALLOC_CAP_SPIRAM);
    }
    if (!eng) {
        ESP_LOGE(TAG, "Failed to allocate memory for KWS engine handle");
        return ESP_ERR_NO_MEM;
    }

    if (config) {
        eng->config = *config;
    } else {
        eng->config = esp_kws_default_config();
    }

    eng->mutex = xSemaphoreCreateMutex();
    if (!eng->mutex) {
        free(eng);
        return ESP_ERR_NO_MEM;
    }

    esp_err_t ret = kws_frontend_init(&eng->frontend);
    if (ret != ESP_OK) {
        vSemaphoreDelete(eng->mutex);
        free(eng);
        return ret;
    }

    ret = kws_nn_init(&eng->nn_state);
    if (ret != ESP_OK) {
        vSemaphoreDelete(eng->mutex);
        free(eng);
        return ret;
    }

    eng->state = eng->config.continuous_streaming ? ESP_KWS_STATE_WAKE_LISTENING : ESP_KWS_STATE_IDLE;
    *out_handle = eng;

    ESP_LOGI(TAG, "KWS Engine initialized (wake threshold: %.2f, cmd threshold: %.2f)",
             eng->config.wake_threshold, eng->config.cmd_threshold);
    return ESP_OK;
}

esp_err_t esp_kws_deinit(esp_kws_handle_t handle) {
    if (!handle) {
        return ESP_ERR_INVALID_ARG;
    }

    xSemaphoreTake(handle->mutex, portMAX_DELAY);
    kws_nn_free_weights(&handle->wake_weights);
    kws_nn_free_weights(&handle->cmd_weights);
    kws_nn_free_state(handle->nn_state);
    xSemaphoreGive(handle->mutex);

    vSemaphoreDelete(handle->mutex);
    free(handle);

    ESP_LOGI(TAG, "KWS Engine deinitialized");
    return ESP_OK;
}

esp_err_t esp_kws_load_wake_model_file(esp_kws_handle_t handle, const char *filepath) {
    if (!handle || !filepath) {
        return ESP_ERR_INVALID_ARG;
    }

    xSemaphoreTake(handle->mutex, portMAX_DELAY);
    kws_nn_free_weights(&handle->wake_weights);
    esp_err_t ret = kws_espm_load_from_file(filepath, &handle->wake_weights);
    handle->wake_model_loaded = (ret == ESP_OK);
    xSemaphoreGive(handle->mutex);

    return ret;
}

esp_err_t esp_kws_load_wake_model_buffer(esp_kws_handle_t handle, const uint8_t *buffer, size_t size) {
    if (!handle || !buffer || size == 0) {
        return ESP_ERR_INVALID_ARG;
    }

    xSemaphoreTake(handle->mutex, portMAX_DELAY);
    kws_nn_free_weights(&handle->wake_weights);
    esp_err_t ret = kws_espm_load_from_buffer(buffer, size, &handle->wake_weights);
    handle->wake_model_loaded = (ret == ESP_OK);
    xSemaphoreGive(handle->mutex);

    return ret;
}

esp_err_t esp_kws_load_cmd_model_file(esp_kws_handle_t handle, const char *filepath) {
    if (!handle || !filepath) {
        return ESP_ERR_INVALID_ARG;
    }

    xSemaphoreTake(handle->mutex, portMAX_DELAY);
    kws_nn_free_weights(&handle->cmd_weights);
    esp_err_t ret = kws_espm_load_from_file(filepath, &handle->cmd_weights);
    handle->cmd_model_loaded = (ret == ESP_OK);
    xSemaphoreGive(handle->mutex);

    return ret;
}

esp_err_t esp_kws_load_cmd_model_buffer(esp_kws_handle_t handle, const uint8_t *buffer, size_t size) {
    if (!handle || !buffer || size == 0) {
        return ESP_ERR_INVALID_ARG;
    }

    xSemaphoreTake(handle->mutex, portMAX_DELAY);
    kws_nn_free_weights(&handle->cmd_weights);
    esp_err_t ret = kws_espm_load_from_buffer(buffer, size, &handle->cmd_weights);
    handle->cmd_model_loaded = (ret == ESP_OK);
    xSemaphoreGive(handle->mutex);

    return ret;
}

esp_err_t esp_kws_register_callback(esp_kws_handle_t handle, esp_kws_callback_t callback, void *user_data) {
    if (!handle) {
        return ESP_ERR_INVALID_ARG;
    }
    xSemaphoreTake(handle->mutex, portMAX_DELAY);
    handle->callback = callback;
    handle->callback_ctx = user_data;
    xSemaphoreGive(handle->mutex);
    return ESP_OK;
}

esp_err_t esp_kws_start_listening(esp_kws_handle_t handle) {
    if (!handle) {
        return ESP_ERR_INVALID_ARG;
    }
    xSemaphoreTake(handle->mutex, portMAX_DELAY);
    if (handle->state == ESP_KWS_STATE_IDLE) {
        handle->state = ESP_KWS_STATE_WAKE_LISTENING;
    }
    xSemaphoreGive(handle->mutex);
    return ESP_OK;
}

esp_err_t esp_kws_start_command_listening(esp_kws_handle_t handle) {
    if (!handle) {
        return ESP_ERR_INVALID_ARG;
    }
    xSemaphoreTake(handle->mutex, portMAX_DELAY);
    handle->state = ESP_KWS_STATE_COMMAND_LISTENING;
    handle->cmd_start_time_us = esp_timer_get_time();
    xSemaphoreGive(handle->mutex);
    return ESP_OK;
}

esp_err_t esp_kws_stop_listening(esp_kws_handle_t handle) {
    if (!handle) {
        return ESP_ERR_INVALID_ARG;
    }
    xSemaphoreTake(handle->mutex, portMAX_DELAY);
    if (!handle->config.continuous_streaming) {
        handle->state = ESP_KWS_STATE_IDLE;
        handle->hop_buf_len = 0;
        kws_nn_reset(handle->nn_state);
        kws_frontend_reset(&handle->frontend);
    }
    xSemaphoreGive(handle->mutex);
    return ESP_OK;
}

esp_kws_state_t esp_kws_get_state(esp_kws_handle_t handle) {
    if (!handle) return ESP_KWS_STATE_UNINITIALIZED;
    return handle->state;
}

esp_err_t esp_kws_reset(esp_kws_handle_t handle) {
    if (!handle) {
        return ESP_ERR_INVALID_ARG;
    }
    xSemaphoreTake(handle->mutex, portMAX_DELAY);
    handle->hop_buf_len = 0;
    handle->state = handle->config.continuous_streaming ? ESP_KWS_STATE_WAKE_LISTENING : ESP_KWS_STATE_IDLE;
    kws_nn_reset(handle->nn_state);
    kws_frontend_reset(&handle->frontend);
    xSemaphoreGive(handle->mutex);
    return ESP_OK;
}

void esp_kws_debug_dump(esp_kws_handle_t handle) {
    if (!handle || !handle->nn_state) return;
    printf("  [KWS DUMP] Mel[0..4]: %.2f, %.2f, %.2f, %.2f, %.2f\n",
           handle->nn_state->mel_history[0 * 64 + 63],
           handle->nn_state->mel_history[1 * 64 + 63],
           handle->nn_state->mel_history[2 * 64 + 63],
           handle->nn_state->mel_history[3 * 64 + 63],
           handle->nn_state->mel_history[4 * 64 + 63]);
    printf("  [KWS DUMP] Pooled[0..4]: %.2f, %.2f, %.2f, %.2f, %.2f\n",
           handle->nn_state->pooled[0],
           handle->nn_state->pooled[1],
           handle->nn_state->pooled[2],
           handle->nn_state->pooled[3],
           handle->nn_state->pooled[4]);
    printf("  [KWS DUMP] Logits: [0]=%.2f, [1]=%.2f\n",
           handle->nn_state->logits[0],
           handle->nn_state->logits[1]);
}

static void process_single_hop(struct esp_kws_engine_s *eng, esp_kws_result_t *out_result) {
    float mel_40[KWS_NUM_MEL_BINS];
    esp_err_t ret = kws_frontend_process_frame(&eng->frontend, eng->hop_buffer, mel_40);
    if (ret != ESP_OK) {
        return;
    }

    kws_nn_push_mel_frame(eng->nn_state, mel_40);

    // Guard: Only evaluate once the receptive temporal window has accumulated frames
    if (eng->nn_state->head_frame < KWS_TIME_WINDOW_FRAMES) {
        return;
    }

    // Check timeout if in command listening state
    if (eng->state == ESP_KWS_STATE_COMMAND_LISTENING) {
        int64_t now_us = esp_timer_get_time();
        int64_t elapsed_ms = (now_us - eng->cmd_start_time_us) / 1000;
        if (elapsed_ms > (int64_t)eng->config.cmd_timeout_ms) {
            ESP_LOGI(TAG, "Command listening window timed out (%lld ms) -> returning to IDLE", (long long)elapsed_ms);
            eng->state = eng->config.continuous_streaming ? ESP_KWS_STATE_WAKE_LISTENING : ESP_KWS_STATE_IDLE;
            return;
        }
    }

    // 1. Wake Word Evaluation
    if ((eng->state == ESP_KWS_STATE_WAKE_LISTENING ||
        (eng->config.continuous_streaming && eng->state == ESP_KWS_STATE_IDLE)) &&
        eng->wake_model_loaded) {

        int64_t t0 = esp_timer_get_time();
        float probs[ESP_KWS_MAX_CLASSES];
        uint16_t best_idx = 0;
        float confidence = kws_nn_forward(&eng->wake_weights, eng->nn_state, probs, &best_idx);
        float latency_ms = (float)(esp_timer_get_time() - t0) / 1000.0f;

        uint16_t n_classes = eng->wake_weights.num_classes;
        if (n_classes > ESP_KWS_MAX_CLASSES) n_classes = ESP_KWS_MAX_CLASSES;

        // Class index 1 is target wake word (0 is background/silence)
        if (best_idx == 1 && confidence >= eng->config.wake_threshold) {
            ESP_LOGI(TAG, "WAKE DETECTED! (confidence: %.2f, latency: %.2f ms)", confidence, latency_ms);

            esp_kws_result_t res = {
                .wake_detected = true,
                .command_detected = false,
                .class_index = best_idx,
                .confidence = confidence,
                .inference_latency_ms = latency_ms,
                .num_classes = n_classes,
            };
            memcpy(res.all_probs, probs, n_classes * sizeof(float));
            snprintf(res.label, sizeof(res.label), "WAKE");

            if (eng->cmd_model_loaded) {
                eng->state = ESP_KWS_STATE_COMMAND_LISTENING;
                eng->cmd_start_time_us = esp_timer_get_time();
            } else {
                eng->state = eng->config.continuous_streaming ? ESP_KWS_STATE_WAKE_LISTENING : ESP_KWS_STATE_IDLE;
            }

            if (out_result) {
                *out_result = res;
            }
            if (eng->callback) {
                eng->callback(&res, eng->callback_ctx);
            }
        } else if (out_result) {
            out_result->wake_detected = false;
            out_result->command_detected = false;
            out_result->class_index = best_idx;
            out_result->confidence = confidence;
            out_result->inference_latency_ms = latency_ms;
            out_result->num_classes = n_classes;
            memcpy(out_result->all_probs, probs, n_classes * sizeof(float));
        }
    }
    // 2. Command Word Evaluation
    else if (eng->state == ESP_KWS_STATE_COMMAND_LISTENING && eng->cmd_model_loaded) {
        int64_t t0 = esp_timer_get_time();
        float probs[ESP_KWS_MAX_CLASSES];
        uint16_t best_idx = 0;
        float confidence = kws_nn_forward(&eng->cmd_weights, eng->nn_state, probs, &best_idx);
        float latency_ms = (float)(esp_timer_get_time() - t0) / 1000.0f;
        uint16_t n_classes = eng->cmd_weights.num_classes;
        if (n_classes > ESP_KWS_MAX_CLASSES) n_classes = ESP_KWS_MAX_CLASSES;

        // Class index > 0 are valid commands (0 is background/noise)
        if (best_idx > 0 && confidence >= eng->config.cmd_threshold) {
            ESP_LOGI(TAG, "COMMAND DETECTED: #%u (confidence: %.2f, latency: %.2f ms)",
                     best_idx, confidence, latency_ms);

            esp_kws_result_t res = {
                .wake_detected = false,
                .command_detected = true,
                .class_index = best_idx,
                .confidence = confidence,
                .inference_latency_ms = latency_ms,
                .num_classes = n_classes,
            };
            memcpy(res.all_probs, probs, n_classes * sizeof(float));
            snprintf(res.label, sizeof(res.label), "CMD_%u", best_idx);

            // Command detected -> return to IDLE / wake listen
            eng->state = eng->config.continuous_streaming ? ESP_KWS_STATE_WAKE_LISTENING : ESP_KWS_STATE_IDLE;

            if (out_result) {
                *out_result = res;
            }
            if (eng->callback) {
                eng->callback(&res, eng->callback_ctx);
            }
        } else if (out_result) {
            out_result->wake_detected = false;
            out_result->command_detected = false;
            out_result->class_index = best_idx;
            out_result->confidence = confidence;
            out_result->inference_latency_ms = latency_ms;
            out_result->num_classes = n_classes;
            memcpy(out_result->all_probs, probs, n_classes * sizeof(float));
        }
    }
}

esp_err_t esp_kws_feed_audio(
    esp_kws_handle_t handle,
    const int16_t *samples,
    size_t num_samples,
    esp_kws_result_t *out_result
) {
    if (!handle || !samples) {
        return ESP_ERR_INVALID_ARG;
    }

    xSemaphoreTake(handle->mutex, portMAX_DELAY);

    size_t in_idx = 0;
    while (in_idx < num_samples) {
        size_t needed = ESP_KWS_FRAME_HOP_SAMPLES - handle->hop_buf_len;
        size_t available = num_samples - in_idx;
        size_t to_copy = (available < needed) ? available : needed;

        memcpy(&handle->hop_buffer[handle->hop_buf_len], &samples[in_idx], to_copy * sizeof(int16_t));
        handle->hop_buf_len += to_copy;
        in_idx += to_copy;

        if (handle->hop_buf_len == ESP_KWS_FRAME_HOP_SAMPLES) {
            process_single_hop(handle, out_result);
            handle->hop_buf_len = 0;
        }
    }

    xSemaphoreGive(handle->mutex);
    return ESP_OK;
}
