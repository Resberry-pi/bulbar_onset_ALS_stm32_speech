#include "nnmf.h"
#include "audio_config.h"
#include "arm_math.h"
#include <math.h>

void nnmf_rank1(float TFM[NUM_FRAMES][NUM_FREQ_BINS],
                float W[NUM_FREQ_BINS],
                float H[NUM_FRAMES])
{
    float prev_error = 0.0f;
    float epsilon = 1e-6f;

    // Initialize W and H with small positive values
    for (int m = 0; m < NUM_FREQ_BINS; m++) W[m] = 0.5f;
    for (int n = 0; n < NUM_FRAMES; n++) H[n] = 0.5f;

    for (int iter = 0; iter < NNMF_ITER; iter++) {
        // Update H
        for (int n = 0; n < NUM_FRAMES; n++) {
            float num = 0.0f, den = 0.0f;
            for (int m = 0; m < NUM_FREQ_BINS; m++) {
                num += W[m] * TFM[n][m];
                den += W[m] * W[m] * H[n];
            }
            if (den > epsilon) H[n] *= num / den;
            if (H[n] < epsilon) H[n] = epsilon;
        }

        // Update W
        for (int m = 0; m < NUM_FREQ_BINS; m++) {
            float num = 0.0f, den = 0.0f;
            for (int n = 0; n < NUM_FRAMES; n++) {
                num += H[n] * TFM[n][m];
                den += H[n] * H[n] * W[m];
            }
            if (den > epsilon) W[m] *= num / den;
            if (W[m] < epsilon) W[m] = epsilon;
        }

        // Optional convergence check
        float reconstruction_error = 0.0f;
        for (int n = 0; n < NUM_FRAMES; n++) {
            for (int m = 0; m < NUM_FREQ_BINS; m++) {
                float approx = W[m] * H[n];
                float diff = TFM[n][m] - approx;
                reconstruction_error += diff * diff;
            }
        }
        if (iter > 0 && fabsf(prev_error - reconstruction_error) < 1e-3f) {
            break;
        }
        prev_error = reconstruction_error;
    }

    //  Removed normalization step
}

void nnmf_rank1_iter(float TFM[NUM_FRAMES][NUM_FREQ_BINS],
                     float W[NUM_FREQ_BINS],
                     float H[NUM_FRAMES],
                     int max_iter)
{
    float prev_error = 0.0f;
    float epsilon = 1e-6f;

    // Initialize W and H
    for (int m = 0; m < NUM_FREQ_BINS; m++) W[m] = 0.5f;
    for (int n = 0; n < NUM_FRAMES; n++) H[n] = 0.5f;

    for (int iter = 0; iter < max_iter; iter++) {
        // Update H
        for (int n = 0; n < NUM_FRAMES; n++) {
            float num = 0.0f, den = 0.0f;
            for (int m = 0; m < NUM_FREQ_BINS; m++) {
                num += W[m] * TFM[n][m];
                den += W[m] * W[m] * H[n];
            }
            if (den > epsilon) H[n] *= num / den;
            if (H[n] < epsilon) H[n] = epsilon;
        }

        // Update W
        for (int m = 0; m < NUM_FREQ_BINS; m++) {
            float num = 0.0f, den = 0.0f;
            for (int n = 0; n < NUM_FRAMES; n++) {
                num += H[n] * TFM[n][m];
                den += H[n] * H[n] * W[m];
            }
            if (den > epsilon) W[m] *= num / den;
            if (W[m] < epsilon) W[m] = epsilon;
        }

        // Optional convergence check
        float reconstruction_error = 0.0f;
        for (int n = 0; n < NUM_FRAMES; n++) {
            for (int m = 0; m < NUM_FREQ_BINS; m++) {
                float approx = W[m] * H[n];
                float diff = TFM[n][m] - approx;
                reconstruction_error += diff * diff;
            }
        }
        if (iter > 0 && fabsf(prev_error - reconstruction_error) < 1e-3f) {
            break;
        }
        prev_error = reconstruction_error;
    }

    //  No normalization applied here either
}


   /* // Normalize
    float w_norm = 0.0f;
    arm_dot_prod_f32(W, W, NUM_FREQ_BINS, &w_norm);
    w_norm = sqrtf(w_norm);
    if (w_norm > epsilon) {
        for (int m = 0; m < NUM_FREQ_BINS; m++) W[m] /= w_norm;
        for (int n = 0; n < NUM_FRAMES; n++) H[n] *= w_norm;
    }
}

// --- Mean of Mean Frequency-Time Frame Relevance ---
float compute_momftfr(const float *H, int len) {
    float weighted_sum = 0.0f;
    float total = 0.0f;
    for (int i = 0; i < len; i++) {
        weighted_sum += i * H[i];
        total += H[i];
    }
    return (total > 0.0f) ? (weighted_sum / total) : 0.0f;
}*/
