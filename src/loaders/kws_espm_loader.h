/**
 * @file kws_espm_loader.h
 * @brief ESPM v3 Container Loader Interface for KWS Engine.
 */

#pragma once

#include "esp_err.h"
#include "kws_nn.h"
#include "esp_model_format.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Load and unpack a KWS model from an in-memory ESPM container buffer.
 *
 * @param buffer Pointer to ESPM file buffer.
 * @param size Size in bytes of buffer.
 * @param out_weights Output weights struct to populate.
 * @return ESP_OK on success.
 */
esp_err_t kws_espm_load_from_buffer(const uint8_t *buffer, size_t size, kws_nn_weights_t *out_weights);

/**
 * @brief Load a KWS model from a file path via VFS (e.g. SD card or SPIFFS).
 *
 * @param filepath Path to .espm file (e.g. "/sd/models/wake_ghost.espm").
 * @param out_weights Output weights struct to populate.
 * @return ESP_OK on success.
 */
esp_err_t kws_espm_load_from_file(const char *filepath, kws_nn_weights_t *out_weights);

#ifdef __cplusplus
}
#endif
