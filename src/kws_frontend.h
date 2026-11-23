/**
 * @file kws_frontend.h
 * @brief 16 kHz Log-Mel Spectrogram Frontend for KWS Engine.
 */

#pragma once

#include "esp_err.h"
#include "esp_kws_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KWS_NUM_MEL_BINS           40
#define KWS_FFT_SIZE               ESP_KWS_N_FFT
#define KWS_WIN_SIZE               ESP_KWS_WIN_LENGTH_SAMPLES   /* 480 samples (30ms) */
#define KWS_HOP_SIZE               ESP_KWS_FRAME_HOP_SAMPLES    /* 160 samples (10ms) */
#define KWS_PREEMPHASIS_COEFF      0.97f
#define KWS_LOG_EPSILON            1e-5f

typedef struct {
    uint16_t start_bin;
    uint16_t num_bins;
    float weights[32];
} kws_sparse_mel_bin_t;

typedef struct {
    float window[KWS_WIN_SIZE];
    float audio_history[KWS_WIN_SIZE];
    float fft_buf[KWS_FFT_SIZE] __attribute__((aligned(16)));
    float power_spectrum[KWS_FFT_SIZE / 2 + 1];
    float prev_sample;
    bool initialized;
} kws_frontend_t;

esp_err_t kws_frontend_init(kws_frontend_t *fe);
void kws_frontend_reset(kws_frontend_t *fe);
esp_err_t kws_frontend_process_frame(kws_frontend_t *fe, const int16_t *pcm_hop, float *out_mel_40);

#ifdef __cplusplus
}
#endif
