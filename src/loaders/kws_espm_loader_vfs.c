/**
 * @file kws_espm_loader_vfs.c
 * @brief VFS / File System Loader for KWS ESPM Models.
 */

#include "kws_espm_loader.h"
#include "esp_log.h"
#include <stdio.h>
#include <stdlib.h>

static const char *TAG = "KWS_ESPM_VFS";

esp_err_t kws_espm_load_from_file(const char *filepath, kws_nn_weights_t *out_weights) {
    if (!filepath || !out_weights) {
        return ESP_ERR_INVALID_ARG;
    }

    FILE *f = fopen(filepath, "rb");
    if (!f) {
        ESP_LOGE(TAG, "Cannot open model file '%s'", filepath);
        return ESP_ERR_NOT_FOUND;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return ESP_ERR_INVALID_SIZE;
    }

    long file_size = ftell(f);
    if (file_size <= 0 || file_size > 32 * 1024 * 1024) { // 32 MB sanity limit
        fclose(f);
        ESP_LOGE(TAG, "Invalid model file size %ld bytes", file_size);
        return ESP_ERR_INVALID_SIZE;
    }

    rewind(f);

    uint8_t *file_buf = (uint8_t *)malloc((size_t)file_size);
    if (!file_buf) {
        fclose(f);
        ESP_LOGE(TAG, "Failed to allocate %ld bytes to read model file", file_size);
        return ESP_ERR_NO_MEM;
    }

    size_t read_bytes = fread(file_buf, 1, (size_t)file_size, f);
    fclose(f);

    if (read_bytes != (size_t)file_size) {
        free(file_buf);
        ESP_LOGE(TAG, "Read mismatch: expected %ld, got %zu", file_size, read_bytes);
        return ESP_FAIL;
    }

    esp_err_t ret = kws_espm_load_from_buffer(file_buf, (size_t)file_size, out_weights);
    free(file_buf);

    return ret;
}
