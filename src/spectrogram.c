#include "spectrogram.h"
#include <math.h>
#include <string.h>

void compute_dynamic_spectrogram(const float32_t *input,
                                 uint32_t length,
                                 uint32_t frame_size,
                                 uint32_t frame_hop,
                                 uint32_t fft_size,
                                 float32_t output[MAX_FRAMES][MAX_BINS],
                                 uint32_t *num_frames_out) {

    uint32_t num_frames = (length - frame_size) / frame_hop + 1;
    if (num_frames > MAX_FRAMES) num_frames = MAX_FRAMES;
    *num_frames_out = num_frames;

    uint32_t num_bins = fft_size / 2;

    arm_rfft_fast_instance_f32 fft;
    arm_rfft_fast_init_f32(&fft, fft_size);

    float32_t window[MAX_FRAME_SIZE];
    float32_t frame[MAX_FFT_SIZE];
    float32_t fft_out[MAX_FFT_SIZE];

    // Hamming window
    for (uint32_t i = 0; i < frame_size; i++) {
        window[i] = 0.54f - 0.46f * cosf((2.0f * PI * i) / (frame_size - 1));
    }

    // Frame-by-frame processing
    for (uint32_t f = 0; f < num_frames; f++) {
        uint32_t offset = f * frame_hop;

        // Apply window
        for (uint32_t i = 0; i < frame_size; i++) {
            frame[i] = input[offset + i] * window[i];
        }

        // Zero-padding if frame_size < fft_size
        for (uint32_t i = frame_size; i < fft_size; i++) {
            frame[i] = 0.0f;
        }

        // FFT
        arm_rfft_fast_f32(&fft, frame, fft_out, 0);

        // Compute raw magnitude (no normalization, no log)
        for (uint32_t i = 0; i < num_bins; i++) {
            float32_t real = fft_out[2 * i];
            float32_t imag = fft_out[2 * i + 1];
            output[f][i] = sqrtf(real * real + imag * imag);
        }
    }
}

// Dummy implementation to preserve linker compatibility
void optimize_spectrogram_for_classifier(const float32_t input[MAX_FRAMES][MAX_BINS],
                                         float32_t output[MAX_FRAMES][MAX_BINS],
                                         uint32_t num_frames,
                                         uint32_t num_bins,
                                         bool apply_smoothing) {
    // Simply copy input to output, do nothing
    for (uint32_t t = 0; t < num_frames; t++) {
        for (uint32_t f = 0; f < num_bins; f++) {
            output[t][f] = input[t][f];
        }
    }
}
