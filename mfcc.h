#ifndef MFCC_H
#define MFCC_H

#include <stdint.h>
#include "arm_math.h"

void pre_emphasis(const float32_t* input, float32_t* output, uint32_t frame_length);


void mfcc_hamming_window(float32_t *input, float32_t *output, uint32_t frame_length);


void mfcc_fft(float32_t *input, float32_t *power_spectrum, uint32_t fft_size);


void mfcc_mel_filterbank(float32_t *power_spectrum, float32_t *mel_energy,
                         uint32_t fft_size, uint32_t num_filters, float32_t sample_rate);


void mfcc_log(float32_t *mel_energy, float32_t *log_energy, uint32_t num_filters);



void mfcc_DCT(float32_t *log_energy, float32_t *output,
              uint32_t num_log_energy, uint32_t num_output);


void compute_mfcc(float* input, int length, float* mfcc_out);


#endif // MFCC_H
