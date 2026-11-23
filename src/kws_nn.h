/**
 * @file kws_nn.h
 * @brief 100% Native C Inference Engine for Streaming TC-ResNet on RISC-V.
 */

#pragma once

#include "esp_err.h"
#include "esp_kws_types.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KWS_TIME_WINDOW_FRAMES      64      /* 64 frames = 640 ms receptive context */
#define KWS_CHANNELS_C              16      /* Base channel depth */
#define KWS_CHANNELS_2C             32      /* Expanded channel depth */

/**
 * @brief Quantized / FP32 Folded Layer Weights Container
 */
typedef struct {
    uint16_t num_classes;
    char labels[ESP_KWS_MAX_CLASSES][ESP_KWS_MAX_LABEL_LEN];

    /* conv0: 40 -> 16 (kernel 3) */
    float *w_conv0;
    float *b_conv0;

    /* block 1: 16 -> 16 */
    float *w_b1_c1;
    float *b_b1_c1;
    float *w_b1_c2;
    float *b_b1_c2;

    /* block 2: 16 -> 32 */
    float *w_b2_c1;
    float *b_b2_c1;
    float *w_b2_c2;
    float *b_b2_c2;
    float *w_b2_skip;
    float *b_b2_skip;

    /* block 3: 32 -> 32 */
    float *w_b3_c1;
    float *b_b3_c1;
    float *w_b3_c2;
    float *b_b3_c2;

    /* dense classifier: 32 -> num_classes */
    float *w_dense;
    float *b_dense;

    /* Memory buffer backing all weights */
    void *allocated_weights_buffer;
    size_t allocated_size;
} kws_nn_weights_t;

/**
 * @brief Streaming Working Buffers (allocated in SRAM / PSRAM)
 */
typedef struct {
    /* Input context ring: 40 mel channels x 64 frames */
    float mel_history[KWS_NUM_MEL_BINS * KWS_TIME_WINDOW_FRAMES];
    uint16_t head_frame;

    /* Intermediate feature maps (triple buffering for zero-clobber residual paths) */
    float act_c1[KWS_CHANNELS_C * KWS_TIME_WINDOW_FRAMES];
    float act_c2[KWS_CHANNELS_C * KWS_TIME_WINDOW_FRAMES];
    float act_c3[KWS_CHANNELS_C * KWS_TIME_WINDOW_FRAMES];
    float act_2c1[KWS_CHANNELS_2C * KWS_TIME_WINDOW_FRAMES];
    float act_2c2[KWS_CHANNELS_2C * KWS_TIME_WINDOW_FRAMES];
    float act_2c3[KWS_CHANNELS_2C * KWS_TIME_WINDOW_FRAMES];

    /* Pooled features */
    float pooled[KWS_CHANNELS_2C];

    /* Logits & Softmax probabilities */
    float logits[ESP_KWS_MAX_CLASSES];
    float probs[ESP_KWS_MAX_CLASSES];
} kws_nn_state_t;

esp_err_t kws_nn_init(kws_nn_state_t **out_state);
void kws_nn_reset(kws_nn_state_t *state);
void kws_nn_free_state(kws_nn_state_t *state);
void kws_nn_free_weights(kws_nn_weights_t *weights);

/**
 * @brief Push one 40-mel frame into the history ring buffer.
 */
void kws_nn_push_mel_frame(kws_nn_state_t *state, const float *mel_40);

/**
 * @brief Execute forward pass inference over current history window.
 *
 * @param weights Loaded model weights.
 * @param state Working inference state.
 * @param out_probs Output probabilities array (size >= weights->num_classes).
 * @param out_best_idx Output index of highest probability class.
 * @return float Highest class confidence (0.0 to 1.0).
 */
float kws_nn_forward(const kws_nn_weights_t *weights, kws_nn_state_t *state, float *out_probs, uint16_t *out_best_idx);

#ifdef __cplusplus
}
#endif
