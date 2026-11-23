/**
 * @file esp_kws.h
 * @brief Public API for ESP-IDF Keyword Spotting (KWS) and Command Word Engine.
 */

#pragma once

#include "esp_err.h"
#include "esp_kws_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct esp_kws_engine_s* esp_kws_handle_t;

/**
 * @brief Callback function invoked when a wake or command keyword is detected.
 */
typedef void (*esp_kws_callback_t)(const esp_kws_result_t *result, void *user_data);

/**
 * @brief Initialize a new KWS engine instance.
 *
 * @param config Configuration pointer (NULL uses recommended defaults).
 * @param out_handle Pointer to receive the allocated engine handle.
 * @return ESP_OK on success, or appropriate error code.
 */
esp_err_t esp_kws_init(const esp_kws_config_t *config, esp_kws_handle_t *out_handle);

/**
 * @brief Deinitialize and release all resources of a KWS engine instance.
 */
esp_err_t esp_kws_deinit(esp_kws_handle_t handle);

/**
 * @brief Load a wake word model from an ESPM v3 file on a mounted VFS (e.g. SD card or SPIFFS).
 */
esp_err_t esp_kws_load_wake_model_file(esp_kws_handle_t handle, const char *filepath);

/**
 * @brief Load a wake word model directly from a memory buffer containing an ESPM v3 container.
 */
esp_err_t esp_kws_load_wake_model_buffer(esp_kws_handle_t handle, const uint8_t *buffer, size_t size);

/**
 * @brief Load a command words model from an ESPM v3 file on a mounted VFS.
 */
esp_err_t esp_kws_load_cmd_model_file(esp_kws_handle_t handle, const char *filepath);

/**
 * @brief Load a command words model directly from a memory buffer.
 */
esp_err_t esp_kws_load_cmd_model_buffer(esp_kws_handle_t handle, const uint8_t *buffer, size_t size);

/**
 * @brief Register a callback for asynchronous detection events.
 */
esp_err_t esp_kws_register_callback(esp_kws_handle_t handle, esp_kws_callback_t callback, void *user_data);

/**
 * @brief Start listening / activate wake or command window (e.g. triggered by VAD onset).
 */
esp_err_t esp_kws_start_listening(esp_kws_handle_t handle);

/**
 * @brief Stop listening / reset active window (e.g. on VAD silence hangover).
 */
esp_err_t esp_kws_stop_listening(esp_kws_handle_t handle);

/**
 * @brief Feed 16 kHz 16-bit mono PCM audio samples into the engine.
 *
 * The audio is buffered into frames (160 samples hop / 10 ms). When sufficient samples
 * are available, feature extraction and streaming neural inference are executed.
 *
 * @param handle KWS engine handle.
 * @param samples Pointer to 16 kHz 16-bit signed PCM audio samples.
 * @param num_samples Number of samples.
 * @param out_result Optional pointer to receive immediate detection result if a frame triggered.
 * @return ESP_OK on success.
 */
esp_err_t esp_kws_feed_audio(esp_kws_handle_t handle, const int16_t *samples, size_t num_samples, esp_kws_result_t *out_result);

/**
 * @brief Manually transition the engine into command listening state.
 * Useful for button triggers or active dialogues.
 */
esp_err_t esp_kws_start_command_listening(esp_kws_handle_t handle);

/**
 * @brief Get the current engine operational state.
 */
esp_kws_state_t esp_kws_get_state(esp_kws_handle_t handle);

/**
 * @brief Reset the internal streaming state (ring buffers and temporal conv history).
 */
esp_err_t esp_kws_reset(esp_kws_handle_t handle);

/**
 * @brief Dump internal state for diagnostics.
 */
void esp_kws_debug_dump(esp_kws_handle_t handle);

#ifdef __cplusplus
}
#endif
