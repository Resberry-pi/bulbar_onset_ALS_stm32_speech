#include "gfcc.h"
#include <math.h>

#define PI 3.14159265359f


/* ================= INIT ================= */

void GFCC_Init(GFCC_Handle *h)
{
    cochlea_init();

    /* ---------- Hamming window ---------- */

    for(int i = 0; i < FRAME_SIZE; i++)
    {
        h->window[i] =
            0.54f - 0.46f *
            cosf((2.0f * PI * i) / (FRAME_SIZE - 1));
    }

    /* ---------- DCT-II matrix ---------- */

    for(int k = 0; k < GFCC_NUM_COEFFS; k++)
    {
        float scale;

        if(k == 0)
            scale = sqrtf(1.0f / GFCC_NUM_FILTERS);
        else
            scale = sqrtf(2.0f / GFCC_NUM_FILTERS);

        for(int i = 0; i < GFCC_NUM_FILTERS; i++)
        {
            h->dct_matrix[k * GFCC_NUM_FILTERS + i] =
                scale *
                cosf((PI * k * (2.0f*i + 1)) /
                     (2.0f * GFCC_NUM_FILTERS));
        }
    }
}


/* ================= PROCESS ================= */

void GFCC_Process(GFCC_Handle *h,
                  float *frame,
                  float *output_features)
{
    /* ---------- DC removal ---------- */

    float mean = 0.0f;

    for(int i = 0; i < FRAME_SIZE; i++)
        mean += frame[i];

    mean /= FRAME_SIZE;


    /* ---------- Windowing ---------- */

    for(int i = 0; i < FRAME_SIZE; i++)
    {
        h->temp_frame[i] =
            (frame[i] - mean) *
            h->window[i];
    }


    /* ---------- Cochlea filterbank ---------- */

    cochlea_process(h->temp_frame,
                    FRAME_SIZE,
                    h->cochlea_out);


    /* ---------- Log Energy (FIXED) ---------- */

    float total_energy = 0.0f;

    for(int i = 0; i < GFCC_NUM_FILTERS; i++)
    {
        float energy = h->cochlea_out[i];

        if(energy < GFCC_EPSILON)
            energy = GFCC_EPSILON;

        /*  keep full dynamic range (NO cbrt) */
        float log_e = logf(energy);

        h->log_energy[i] = log_e;

        total_energy += energy;
    }


    /* ---------- DCT ---------- */

    for(int k = 0; k < GFCC_NUM_COEFFS; k++)
    {
        float sum = 0.0f;

        for(int i = 0; i < GFCC_NUM_FILTERS; i++)
        {
            sum +=
                h->log_energy[i] *
                h->dct_matrix[k * GFCC_NUM_FILTERS + i];
        }

        h->gfcc[k] = sum;
        output_features[k] = sum;
    }


    /* ---------- Optional: Stronger C0 ---------- */

    h->gfcc[0] = logf(total_energy + GFCC_EPSILON);
    output_features[0] = h->gfcc[0];
}
