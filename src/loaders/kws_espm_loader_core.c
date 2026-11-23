/**
 * @file kws_espm_loader_core.c
 * @brief Core Parser and Validator for KWS ESPM v3 Container.
 */

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_model_format.h"
#include "kws_espm_loader.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static const char *TAG = "KWS_ESPM_CORE";

#define EXPECTED_TENSOR_COUNT 18

typedef struct {
  const char *name;
  size_t expected_elements;
  float **target_ptr;
} tensor_binding_t;

static const esp_model_section_t *
find_section(const esp_model_section_t *sections, uint16_t num_sections,
             uint16_t section_type) {
  for (uint16_t i = 0; i < num_sections; i++) {
    if (sections[i].section_type == section_type) {
      return &sections[i];
    }
  }
  return NULL;
}

esp_err_t kws_espm_load_from_buffer(const uint8_t *buffer, size_t size,
                                    kws_nn_weights_t *out_weights) {
  if (!buffer || size < sizeof(esp_model_header_t) || !out_weights) {
    return ESP_ERR_INVALID_ARG;
  }

  const esp_model_header_t *hdr = (const esp_model_header_t *)buffer;

  // 1. Verify Magic and Version
  if (hdr->magic != ESPM_MAGIC_LE) {
    ESP_LOGE(TAG, "Invalid magic 0x%08X (expected ESPM 0x%08X)",
             (unsigned)hdr->magic, (unsigned)ESPM_MAGIC_LE);
    return ESP_ERR_INVALID_STATE;
  }

  if (hdr->format_version != ESPM_CURRENT_VERSION) {
    ESP_LOGE(TAG, "Unsupported format version %u (expected %u)",
             hdr->format_version, ESPM_CURRENT_VERSION);
    return ESP_ERR_NOT_SUPPORTED;
  }

  if (hdr->model_type != MODEL_TYPE_KWS) {
    ESP_LOGE(TAG, "Invalid model type %u (expected MODEL_TYPE_KWS = 2)",
             hdr->model_type);
    return ESP_ERR_INVALID_STATE;
  }

  // 2. Parse Section Table
  size_t sec_table_offset = sizeof(esp_model_header_t);
  size_t sec_table_size = hdr->num_sections * sizeof(esp_model_section_t);
  if (sec_table_offset + sec_table_size > size) {
    ESP_LOGE(TAG, "Section table exceeds file boundary");
    return ESP_ERR_INVALID_SIZE;
  }

  const esp_model_section_t *sections =
      (const esp_model_section_t *)(buffer + sec_table_offset);
  const esp_model_section_t *idx_sec =
      find_section(sections, hdr->num_sections, SECTION_TYPE_TENSOR_INDEX);
  const esp_model_section_t *dat_sec =
      find_section(sections, hdr->num_sections, SECTION_TYPE_TENSOR_DATA);

  if (!idx_sec || !dat_sec) {
    ESP_LOGE(TAG, "Missing TENSOR_INDEX or TENSOR_DATA section");
    return ESP_ERR_NOT_FOUND;
  }

  if (idx_sec->offset + idx_sec->size > size ||
      dat_sec->offset + dat_sec->size > size) {
    ESP_LOGE(TAG, "Tensor section offsets out of file range");
    return ESP_ERR_INVALID_SIZE;
  }

  // 3. Read Index Header (v2 tensor index format)
  const uint8_t *idx_raw = buffer + idx_sec->offset;
  const esp_model_tensor_index_header_v2_t *idx_hdr =
      (const esp_model_tensor_index_header_v2_t *)idx_raw;

  if (memcmp(idx_hdr->magic, ESPM_TENSOR_INDEX_MAGIC, 4) != 0) {
    ESP_LOGE(TAG, "Invalid tensor index magic");
    return ESP_ERR_INVALID_STATE;
  }

  uint32_t num_tensors = idx_hdr->tensor_count;
  if (num_tensors == 0 || num_tensors > ESPM_MAX_TENSORS) {
    ESP_LOGE(TAG, "Invalid tensor count: %u", (unsigned)num_tensors);
    return ESP_ERR_INVALID_SIZE;
  }

  const uint8_t *tensors_start =
      idx_raw + sizeof(esp_model_tensor_index_header_v2_t);
  const esp_model_tensor_record_v2_t *records =
      (const esp_model_tensor_record_v2_t *)tensors_start;

  // Allocate memory copy for weights in fast internal SRAM (fallback to PSRAM)
  size_t data_bytes = (size_t)dat_sec->size;
  float *weights_copy = (float *)heap_caps_malloc(
      data_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!weights_copy) {
    weights_copy = (float *)heap_caps_malloc(data_bytes, MALLOC_CAP_SPIRAM |
                                                             MALLOC_CAP_8BIT);
  }
  if (!weights_copy) {
    ESP_LOGE(TAG, "Failed to allocate %zu bytes for KWS weights", data_bytes);
    return ESP_ERR_NO_MEM;
  }
  memcpy(weights_copy, buffer + dat_sec->offset, data_bytes);

  memset(out_weights, 0, sizeof(kws_nn_weights_t));
  out_weights->allocated_weights_buffer = weights_copy;
  out_weights->allocated_size = data_bytes;

  // Default classes
  out_weights->num_classes =
      2; // Default binary wake word (target vs background)

  // Setup bindings
  tensor_binding_t bindings[] = {
      {"conv0.weight", 40 * 16 * 3, &out_weights->w_conv0},
      {"conv0.bias", 16, &out_weights->b_conv0},
      {"block1.conv1.weight", 16 * 16 * 3, &out_weights->w_b1_c1},
      {"block1.conv1.bias", 16, &out_weights->b_b1_c1},
      {"block1.conv2.weight", 16 * 16 * 3, &out_weights->w_b1_c2},
      {"block1.conv2.bias", 16, &out_weights->b_b1_c2},
      {"block2.conv1.weight", 16 * 32 * 3, &out_weights->w_b2_c1},
      {"block2.conv1.bias", 32, &out_weights->b_b2_c1},
      {"block2.conv2.weight", 32 * 32 * 3, &out_weights->w_b2_c2},
      {"block2.conv2.bias", 32, &out_weights->b_b2_c2},
      {"block2.skip.weight", 16 * 32 * 1, &out_weights->w_b2_skip},
      {"block2.skip.bias", 32, &out_weights->b_b2_skip},
      {"block3.conv1.weight", 32 * 32 * 3, &out_weights->w_b3_c1},
      {"block3.conv1.bias", 32, &out_weights->b_b3_c1},
      {"block3.conv2.weight", 32 * 32 * 3, &out_weights->w_b3_c2},
      {"block3.conv2.bias", 32, &out_weights->b_b3_c2},
      {"classifier.weight", 0, &out_weights->w_dense},
      {"classifier.bias", 0, &out_weights->b_dense},
  };
  size_t num_bindings = sizeof(bindings) / sizeof(bindings[0]);

  for (uint32_t t = 0; t < num_tensors; t++) {
    const esp_model_tensor_record_v2_t *rec = &records[t];
    const char *t_name = rec->name;

    for (size_t b = 0; b < num_bindings; b++) {
      if (strcmp(t_name, bindings[b].name) == 0) {
        size_t offset_bytes = (size_t)rec->data_offset;
        *(bindings[b].target_ptr) =
            (float *)((uint8_t *)weights_copy + offset_bytes);

        // Detect classifier classes
        if (strcmp(t_name, "classifier.bias") == 0) {
          out_weights->num_classes = (uint16_t)rec->shape[0];
        }
        break;
      }
    }
  }

  ESP_LOGI(
      TAG,
      "Successfully loaded KWS ESPM model '%s' (%u classes, %zu KB weights)",
      hdr->model_name, out_weights->num_classes, data_bytes / 1024);

  return ESP_OK;
}
