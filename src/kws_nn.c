/**
 * @file kws_nn.c
 * @brief Optimized Native C Inference Implementation for Streaming TC-ResNet.
 */

#include "kws_nn.h"
#include "kws_frontend.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

static const char *TAG = "KWS_NN";

#define T_FRAMES KWS_TIME_WINDOW_FRAMES
#define SILENCE_LOG_MEL (-11.512925f)

esp_err_t kws_nn_init(kws_nn_state_t **out_state) {
    if (!out_state) {
        return ESP_ERR_INVALID_ARG;
    }

    // Allocate working state in internal SRAM for maximum execution speed
    kws_nn_state_t *st = (kws_nn_state_t *)heap_caps_calloc(1, sizeof(kws_nn_state_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    if (!st) {
        st = (kws_nn_state_t *)heap_caps_calloc(1, sizeof(kws_nn_state_t), MALLOC_CAP_SPIRAM);
    }
    if (!st) {
        ESP_LOGE(TAG, "Failed to allocate memory for KWS NN state (%zu bytes)", sizeof(kws_nn_state_t));
        return ESP_ERR_NO_MEM;
    }

    kws_nn_reset(st);
    *out_state = st;
    return ESP_OK;
}

void kws_nn_reset(kws_nn_state_t *state) {
    if (state) {
        // Initialize ring buffer with acoustic silence floor (-11.51 dB)
        for (int i = 0; i < KWS_NUM_MEL_BINS * T_FRAMES; i++) {
            state->mel_history[i] = SILENCE_LOG_MEL;
        }
        state->head_frame = 0;
    }
}

void kws_nn_free_state(kws_nn_state_t *state) {
    if (state) {
        free(state);
    }
}

void kws_nn_free_weights(kws_nn_weights_t *weights) {
    if (weights) {
        if (weights->allocated_weights_buffer) {
            free(weights->allocated_weights_buffer);
            weights->allocated_weights_buffer = NULL;
        }
        memset(weights, 0, sizeof(kws_nn_weights_t));
    }
}

void kws_nn_push_mel_frame(kws_nn_state_t *state, const float *mel_40) {
    if (!state || !mel_40) {
        return;
    }

    // Shift history left by 1 frame and append new frame at the end
    // Layout: mel_history[m * T_FRAMES + t]
    for (int m = 0; m < KWS_NUM_MEL_BINS; m++) {
        float *row = &state->mel_history[m * T_FRAMES];
        memmove(&row[0], &row[1], (T_FRAMES - 1) * sizeof(float));
        row[T_FRAMES - 1] = mel_40[m];
    }

    if (state->head_frame < T_FRAMES) {
        state->head_frame++;
    }
}

/**
 * High-Performance Cache-Linear 1D Temporal Convolution
 * Optimized for RISC-V FPU pipelining and zero-branch inner loops.
 */
static void conv1d_forward_fast(
    const float *in_data, int c_in, int t_len,
    const float *weights, const float *biases,
    int c_out, int kernel_size, int dilation,
    bool relu, float *out_data
) {
    const int pad = (kernel_size - 1) * dilation / 2;

    for (int co = 0; co < c_out; co++) {
        float *out_row = &out_data[co * t_len];
        const float *w_co = &weights[co * c_in * kernel_size];
        float bias = biases ? biases[co] : 0.0f;

        // 1. Initialize output row with bias
        for (int t = 0; t < t_len; t++) {
            out_row[t] = bias;
        }

        // 2. Kernel size 1 (Shortcut projection / 1x1 conv)
        if (kernel_size == 1) {
            for (int ci = 0; ci < c_in; ci++) {
                const float *in_row = &in_data[ci * t_len];
                float w = w_co[ci];
                for (int t = 0; t < t_len; t++) {
                    out_row[t] += in_row[t] * w;
                }
            }
        }
        // 3. Kernel size 3 (Temporal convolutions with dilation)
        else if (kernel_size == 3) {
            int d = dilation;
            for (int ci = 0; ci < c_in; ci++) {
                const float *in_row = &in_data[ci * t_len];
                const float *w_ci = &w_co[ci * 3];
                float w0 = w_ci[0];
                float w1 = w_ci[1];
                float w2 = w_ci[2];

                // Left border: t < pad
                for (int t = 0; t < pad; t++) {
                    float acc = 0.0f;
                    int t0 = t - d;
                    int t1 = t;
                    int t2 = t + d;
                    if (t0 >= 0) acc += in_row[t0] * w0;
                    acc += in_row[t1] * w1;
                    if (t2 < t_len) acc += in_row[t2] * w2;
                    out_row[t] += acc;
                }

                // Bulk inner loop: pad <= t < t_len - pad (100% branch-free!)
                int t_bulk_end = t_len - pad;
                for (int t = pad; t < t_bulk_end; t++) {
                    out_row[t] += in_row[t - d] * w0 + in_row[t] * w1 + in_row[t + d] * w2;
                }

                // Right border: t >= t_bulk_end
                for (int t = t_bulk_end; t < t_len; t++) {
                    float acc = 0.0f;
                    int t0 = t - d;
                    int t1 = t;
                    int t2 = t + d;
                    if (t0 >= 0) acc += in_row[t0] * w0;
                    acc += in_row[t1] * w1;
                    if (t2 < t_len) acc += in_row[t2] * w2;
                    out_row[t] += acc;
                }
            }
        }

        // 4. In-place ReLU activation if requested
        if (relu) {
            for (int t = 0; t < t_len; t++) {
                if (out_row[t] < 0.0f) {
                    out_row[t] = 0.0f;
                }
            }
        }
    }
}

/**
 * Element-wise addition with optional ReLU
 */
static void add_residual_relu(
    const float *src_a, const float *src_b,
    int length, float *dst
) {
    int i = 0;
    for (; i + 3 < length; i += 4) {
        float v0 = src_a[i + 0] + src_b[i + 0];
        float v1 = src_a[i + 1] + src_b[i + 1];
        float v2 = src_a[i + 2] + src_b[i + 2];
        float v3 = src_a[i + 3] + src_b[i + 3];

        dst[i + 0] = v0 > 0.0f ? v0 : 0.0f;
        dst[i + 1] = v1 > 0.0f ? v1 : 0.0f;
        dst[i + 2] = v2 > 0.0f ? v2 : 0.0f;
        dst[i + 3] = v3 > 0.0f ? v3 : 0.0f;
    }
    for (; i < length; i++) {
        float v = src_a[i] + src_b[i];
        dst[i] = v > 0.0f ? v : 0.0f;
    }
}

float kws_nn_forward(
    const kws_nn_weights_t *weights,
    kws_nn_state_t *state,
    float *out_probs,
    uint16_t *out_best_idx
) {
    if (!weights || !state || !out_probs || weights->num_classes == 0) {
        return 0.0f;
    }

    // 1. Conv0: (40, T) -> (16, T) into act_c1
    conv1d_forward_fast(
        state->mel_history, KWS_NUM_MEL_BINS, T_FRAMES,
        weights->w_conv0, weights->b_conv0,
        KWS_CHANNELS_C, 3, 1, true, state->act_c1
    );

    // 2. Block 1 (16 -> 16): act_c1 is residual shortcut
    conv1d_forward_fast(
        state->act_c1, KWS_CHANNELS_C, T_FRAMES,
        weights->w_b1_c1, weights->b_b1_c1,
        KWS_CHANNELS_C, 3, 1, true, state->act_c2
    );
    conv1d_forward_fast(
        state->act_c2, KWS_CHANNELS_C, T_FRAMES,
        weights->w_b1_c2, weights->b_b1_c2,
        KWS_CHANNELS_C, 3, 1, false, state->act_c3
    );
    // Shortcut add: act_c3 + act_c1 -> act_c1
    add_residual_relu(state->act_c3, state->act_c1, KWS_CHANNELS_C * T_FRAMES, state->act_c1);

    // 3. Block 2 (16 -> 32): act_c1 is residual input
    // Shortcut projection: (16 -> 32) kernel 1 into act_2c3
    conv1d_forward_fast(
        state->act_c1, KWS_CHANNELS_C, T_FRAMES,
        weights->w_b2_skip, weights->b_b2_skip,
        KWS_CHANNELS_2C, 1, 1, false, state->act_2c3
    );
    // Conv 2a: (16 -> 32) kernel 3, dilation 2 into act_2c1
    conv1d_forward_fast(
        state->act_c1, KWS_CHANNELS_C, T_FRAMES,
        weights->w_b2_c1, weights->b_b2_c1,
        KWS_CHANNELS_2C, 3, 2, true, state->act_2c1
    );
    // Conv 2b: (32 -> 32) kernel 3, dilation 1: act_2c1 -> act_2c2
    conv1d_forward_fast(
        state->act_2c1, KWS_CHANNELS_2C, T_FRAMES,
        weights->w_b2_c2, weights->b_b2_c2,
        KWS_CHANNELS_2C, 3, 1, false, state->act_2c2
    );
    // Add shortcut projection (act_2c3) + conv output (act_2c2) -> act_2c1
    add_residual_relu(state->act_2c2, state->act_2c3, KWS_CHANNELS_2C * T_FRAMES, state->act_2c1);

    // 4. Block 3 (32 -> 32): act_2c1 is residual input
    // Conv 3a: act_2c1 -> act_2c2
    conv1d_forward_fast(
        state->act_2c1, KWS_CHANNELS_2C, T_FRAMES,
        weights->w_b3_c1, weights->b_b3_c1,
        KWS_CHANNELS_2C, 3, 2, true, state->act_2c2
    );
    // Conv 3b: act_2c2 -> act_2c3
    conv1d_forward_fast(
        state->act_2c2, KWS_CHANNELS_2C, T_FRAMES,
        weights->w_b3_c2, weights->b_b3_c2,
        KWS_CHANNELS_2C, 3, 1, false, state->act_2c3
    );
    // Add shortcut (act_2c1) + conv output (act_2c3) -> act_2c1
    add_residual_relu(state->act_2c3, state->act_2c1, KWS_CHANNELS_2C * T_FRAMES, state->act_2c1);

    // 5. Global Average Pooling over time
    const float inv_t = 1.0f / (float)T_FRAMES;
    for (int c = 0; c < KWS_CHANNELS_2C; c++) {
        float sum = 0.0f;
        const float *row = &state->act_2c1[c * T_FRAMES];
        for (int t = 0; t < T_FRAMES; t++) {
            sum += row[t];
        }
        state->pooled[c] = sum * inv_t;
    }

    // 6. Dense Linear Classifier: (32) -> (num_classes)
    float max_logit = -1e9f;
    for (uint16_t k = 0; k < weights->num_classes; k++) {
        float acc = weights->b_dense ? weights->b_dense[k] : 0.0f;
        const float *w_k = &weights->w_dense[k * KWS_CHANNELS_2C];

        for (int c = 0; c < KWS_CHANNELS_2C; c++) {
            acc += state->pooled[c] * w_k[c];
        }
        state->logits[k] = acc;
        if (acc > max_logit) {
            max_logit = acc;
        }
    }

    // 7. Stable Softmax
    float sum_exp = 0.0f;
    for (uint16_t k = 0; k < weights->num_classes; k++) {
        float e = expf(state->logits[k] - max_logit);
        state->probs[k] = e;
        sum_exp += e;
    }

    float inv_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1.0f);
    float best_prob = 0.0f;
    uint16_t best_idx = 0;

    for (uint16_t k = 0; k < weights->num_classes; k++) {
        float p = state->probs[k] * inv_sum;
        out_probs[k] = p;
        if (p > best_prob) {
            best_prob = p;
            best_idx = k;
        }
    }

    if (out_best_idx) {
        *out_best_idx = best_idx;
    }

    return best_prob;
}
