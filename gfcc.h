#ifndef GFCC_H
#define GFCC_H

#include <stdint.h>
#include "cochlea.h"

/* Frame parameters */
#define FRAME_SIZE        1024
#define FRAME_HOP         512

/* GFCC parameters */
#define GFCC_NUM_FILTERS  COCHLEA_CHANNELS
#define GFCC_NUM_COEFFS   13
#define GFCC_EPSILON      1e-8f

typedef struct
{
    /* Hamming window */
    float window[FRAME_SIZE];

    /* temporary frame buffer */
    float temp_frame[FRAME_SIZE];

    /* cochlea filterbank output */
    float cochlea_out[GFCC_NUM_FILTERS];

    /* log energy */
    float log_energy[GFCC_NUM_FILTERS];

    /* DCT matrix */
    float dct_matrix[GFCC_NUM_COEFFS * GFCC_NUM_FILTERS];

    /* GFCC output */
    float gfcc[GFCC_NUM_COEFFS];

} GFCC_Handle;


/* initialization */
void GFCC_Init(GFCC_Handle *h);

/* process one frame */
void GFCC_Process(GFCC_Handle *h,
                  float *frame,
                  float *output_features);

#endif
