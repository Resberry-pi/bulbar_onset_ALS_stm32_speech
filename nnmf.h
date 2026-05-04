#ifndef __NNMF_H__
#define __NNMF_H__

#include "audio_config.h"

void nnmf_rank1(float TFM[NUM_FRAMES][NUM_FREQ_BINS],
                float W[NUM_FREQ_BINS],
                float H[NUM_FRAMES]);

void nnmf_rank1_iter(float TFM[NUM_FRAMES][NUM_FREQ_BINS],
                     float W[NUM_FREQ_BINS],
                     float H[NUM_FRAMES],
                     int max_iter);

#endif // __NNMF_H__
