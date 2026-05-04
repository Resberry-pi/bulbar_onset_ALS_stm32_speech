#include "mfcc.h"
#include <math.h>
#include <stdlib.h>

#define PI 3.14159265359f

// Pre-emphasis filter: y[n] = x[n] - pre_emphasis * x[n-1]
void pre_emphasis(const float32_t* input, float32_t* output, uint32_t frame_length) {
    float32_t pre_emphasis = 0.97f;
    output[0] = input[0];
    for (uint32_t i = 1; i < frame_length; i++) {
        output[i] = input[i] - pre_emphasis * input[i - 1];
    }
}

// Hamming window: w[n] = 0.54 - 0.46 * cos(2 * PI * n / (N-1))
void mfcc_hamming_window(float32_t *input, float32_t *output, uint32_t frame_length) {
    for (uint32_t i = 0; i < frame_length; i++) {
        float32_t w = 0.54f - 0.46f * arm_cos_f32((2.0f * PI * i) / (frame_length - 1));
        output[i] = input[i] * w;
    }
}

// Compute FFT and convert to power spectrum
void mfcc_fft(float32_t *input, float32_t *power_spectrum, uint32_t fft_size) {
    arm_rfft_fast_instance_f32 fft_instance;
    float32_t fft_output[fft_size];
    arm_rfft_fast_init_f32(&fft_instance, fft_size);
    arm_rfft_fast_f32(&fft_instance, input, fft_output, 0);

    for (uint32_t i = 0; i < fft_size / 2; i++) {
        float32_t real = fft_output[2 * i];
        float32_t imag = fft_output[2 * i + 1];
        power_spectrum[i] = real * real + imag * imag;
    }
}

// Convert frequency (Hz) to mel scale
static float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

// Convert mel to frequency (Hz)
static float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// Generate Mel filterbank and apply to power spectrum
void mfcc_mel_filterbank(float32_t *power_spectrum, float32_t *mel_energy, uint32_t fft_size,
                         uint32_t num_filters, float32_t sample_rate) {
    uint32_t num_fft_bins = fft_size / 2;
    float mel_min = hz_to_mel(0);
    float mel_max = hz_to_mel(sample_rate / 2);   //reference from stm32 community
    float mel_points[num_filters + 2];
    float hz_points[num_filters + 2];
    uint32_t bin[num_filters + 2];

    for (uint32_t i = 0; i < num_filters + 2; i++) {
        mel_points[i] = mel_min + i * (mel_max - mel_min) / (num_filters + 1);
        hz_points[i] = mel_to_hz(mel_points[i]);
        bin[i] = (uint32_t)(hz_points[i] * fft_size / sample_rate);
    }

    for (uint32_t i = 0; i < num_filters; i++) {
        mel_energy[i] = 0.0f;
        for (uint32_t j = bin[i]; j < bin[i + 1]; j++) {
            mel_energy[i] += ((float32_t)(j - bin[i]) / (bin[i + 1] - bin[i])) * power_spectrum[j];
        }
        for (uint32_t j = bin[i + 1]; j < bin[i + 2]; j++) {
            mel_energy[i] += ((float32_t)(bin[i + 2] - j) / (bin[i + 2] - bin[i + 1])) * power_spectrum[j];
        }
    }
}

// Log compression
void mfcc_log(float32_t *mel_energy, float32_t *log_energy, uint32_t num_filters) {
    for (uint32_t i = 0; i < num_filters; i++) {
        log_energy[i] = logf(mel_energy[i] + 1e-6f);
    }
}

// DCT type-II (naive implementation)
void mfcc_DCT(float32_t *log_energy, float32_t *output, uint32_t num_log_energy, uint32_t num_output) {
    for (uint32_t k = 0; k < num_output; k++) {
        output[k] = 0.0f;
        for (uint32_t n = 0; n < num_log_energy; n++) {
            output[k] += log_energy[n] * cosf(PI * k * (2 * n + 1) / (2.0f * num_log_energy));
        }
    }
}

void compute_mfcc(float* input, int length, float* mfcc_out) {
    const uint32_t frame_length = 1024;
    const uint32_t fft_size = 1024;
    const uint32_t num_filters = 26;
    const uint32_t num_coeffs = 13;
    const float32_t sample_rate = 16000.0f;

    float32_t preemphasized[frame_length];
    float32_t windowed[frame_length];
    float32_t power_spectrum[fft_size / 2];
    float32_t mel_energy[num_filters];
    float32_t log_energy[num_filters];

    // 1. Pre-emphasis
    pre_emphasis(input, preemphasized, frame_length);

    // 2. Hamming window
    mfcc_hamming_window(preemphasized, windowed, frame_length);

    // 3. Power Spectrum using FFT
    mfcc_fft(windowed, power_spectrum, fft_size);

    // 4. Mel filterbank
    mfcc_mel_filterbank(power_spectrum, mel_energy, fft_size, num_filters, sample_rate);

    // 5. Log compression
    mfcc_log(mel_energy, log_energy, num_filters);

    // 6. DCT to get MFCC
    mfcc_DCT(log_energy, mfcc_out, num_filters, num_coeffs);
}
