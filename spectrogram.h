#ifndef __SPECTROGRAM_DYNAMIC_H__
#define __SPECTROGRAM_DYNAMIC_H__

#include <stdint.h>
#include <stdbool.h>
#include "arm_math.h"

#define MAX_FRAMES     30
#define MAX_BINS       512
#define MAX_FRAME_SIZE 1024
#define MAX_FFT_SIZE   1024

void compute_dynamic_spectrogram(const float32_t *input,
                                 uint32_t length,
                                 uint32_t frame_size,
                                 uint32_t frame_hop,
                                 uint32_t fft_size,
                                 float32_t output[MAX_FRAMES][MAX_BINS],
                                 uint32_t *num_frames_out);


#endif
