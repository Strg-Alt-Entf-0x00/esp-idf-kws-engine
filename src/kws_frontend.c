/**
 * @file kws_frontend.c
 * @brief 16 kHz Log-Mel Filterbank Audio Frontend matching Python Bit-for-Bit.
 */

#include "kws_frontend.h"
#include "kws_mel_table.h"
#include "dsps_fft4r.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

esp_err_t kws_frontend_init(kws_frontend_t *fe) {
    if (!fe) {
        return ESP_ERR_INVALID_ARG;
    }

    // Initialize 256-point complex radix-4 FFT (packs 512 real samples)
    esp_err_t ret = dsps_fft4r_init_fc32(NULL, KWS_FFT_SIZE / 2);
    if (ret != ESP_OK) {
        return ret;
    }

    // Compute periodic Hann window: 0.5 - 0.5 * cos(2 * pi * n / N)
    for (int n = 0; n < KWS_WIN_SIZE; n++) {
        fe->window[n] = 0.5f - 0.5f * cosf((2.0f * (float)M_PI * (float)n) / (float)KWS_WIN_SIZE);
    }

    kws_frontend_reset(fe);
    fe->initialized = true;
    return ESP_OK;
}

void kws_frontend_reset(kws_frontend_t *fe) {
    if (fe) {
        memset(fe->audio_history, 0, sizeof(fe->audio_history));
        fe->prev_sample = 0.0f;
    }
}

esp_err_t kws_frontend_process_frame(kws_frontend_t *fe, const int16_t *pcm_hop, float *out_mel_40) {
    if (!fe || !fe->initialized || !pcm_hop || !out_mel_40) {
        return ESP_ERR_INVALID_ARG;
    }

    // 1. Apply pre-emphasis on incoming 160 samples
    float emphasized_hop[KWS_HOP_SIZE];
    for (int i = 0; i < KWS_HOP_SIZE; i++) {
        float cur = (float)pcm_hop[i];
        emphasized_hop[i] = cur - KWS_PREEMPHASIS_COEFF * fe->prev_sample;
        fe->prev_sample = cur;
    }

    // 2. Shift sliding 480-sample history window left by 160 and append new hop
    memmove(&fe->audio_history[0], &fe->audio_history[KWS_HOP_SIZE], (KWS_WIN_SIZE - KWS_HOP_SIZE) * sizeof(float));
    memcpy(&fe->audio_history[KWS_WIN_SIZE - KWS_HOP_SIZE], emphasized_hop, KWS_HOP_SIZE * sizeof(float));

    // 3. Prepare 512-point FFT buffer with zero-padding on edges
    memset(fe->fft_buf, 0, KWS_FFT_SIZE * sizeof(float));

    // Window: 480 samples centered in 512 FFT buffer (16 zeros prefix)
    const int offset = (KWS_FFT_SIZE - KWS_WIN_SIZE) / 2; // 16
    for (int i = 0; i < KWS_WIN_SIZE; i++) {
        fe->fft_buf[offset + i] = fe->audio_history[i] * fe->window[i];
    }

    // 4. Execute Radix-4 Real-FFT via ESP-DSP
    dsps_fft4r_fc32(fe->fft_buf, KWS_FFT_SIZE / 2);
    dsps_bit_rev4r_fc32(fe->fft_buf, KWS_FFT_SIZE / 2);
    dsps_cplx2real_fc32(fe->fft_buf, KWS_FFT_SIZE / 2);

    // 5. Compute Power Spectrum: |X[k]|^2
    // DC bin (k = 0)
    fe->power_spectrum[0] = fe->fft_buf[0] * fe->fft_buf[0];
    // Nyquist bin (k = 256)
    fe->power_spectrum[KWS_FFT_SIZE / 2] = fe->fft_buf[1] * fe->fft_buf[1];

    for (int k = 1; k < KWS_FFT_SIZE / 2; k++) {
        float re = fe->fft_buf[2 * k];
        float im = fe->fft_buf[2 * k + 1];
        fe->power_spectrum[k] = re * re + im * im;
    }

    // 6. Filterbank multiplication & log compression
    for (int m = 0; m < KWS_NUM_MEL_BINS; m++) {
        float energy = 0.0f;
        const kws_sparse_mel_bin_t *fb = &KWS_SPARSE_MEL_BINS[m];
        for (uint16_t b = 0; b < fb->num_bins; b++) {
            energy += fe->power_spectrum[fb->start_bin + b] * fb->weights[b];
        }
        out_mel_40[m] = logf(energy + KWS_LOG_EPSILON);
    }

    return ESP_OK;
}
