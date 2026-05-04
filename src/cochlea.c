#include "cochlea.h"
#include <math.h>

#define SAMPLE_RATE 16000.0f
#define MIN_FREQ    100.0f
#define MAX_FREQ    8000.0f

/*   envelope */
#define ENV_ALPHA   0.2f
#define ENV_DECAY   0.8f

#define PI 3.14159265359f

/* ================= ERB FUNCTIONS ================= */

static float hz_to_erb(float f)
{
    return 21.4f * log10f(4.37f * f / 1000.0f + 1.0f);
}

static float erb_to_hz(float erb)
{
    return (1000.0f / 4.37f) *
           (powf(10.0f, erb / 21.4f) - 1.0f);
}

static float erb_bandwidth(float fc)
{
    return 24.7f * (4.37f * fc / 1000.0f + 1.0f);
}

/* ================= FILTER STORAGE ================= */

static float center_freq[COCHLEA_CHANNELS];

static float a1[COCHLEA_CHANNELS];
static float a2[COCHLEA_CHANNELS];
static float b0[COCHLEA_CHANNELS];

/* ================= FILTER STATE ================= */

static float y1_1[COCHLEA_CHANNELS];
static float y2_1[COCHLEA_CHANNELS];

static float y1_2[COCHLEA_CHANNELS];
static float y2_2[COCHLEA_CHANNELS];

static float env_state[COCHLEA_CHANNELS];

/* ================= INIT ================= */

void cochlea_init(void)
{
    float erb_min = hz_to_erb(MIN_FREQ);
    float erb_max = hz_to_erb(MAX_FREQ);

    for(int i=0;i<COCHLEA_CHANNELS;i++)
    {
        float erb_point = erb_min +
            (erb_max-erb_min)*i/(COCHLEA_CHANNELS-1);

        center_freq[i] = erb_to_hz(erb_point);

        float bw = erb_bandwidth(center_freq[i]);

        float r = expf(-PI*bw/SAMPLE_RATE);
        float theta = 2.0f*PI*center_freq[i]/SAMPLE_RATE;

        a1[i] = -2.0f*r*cosf(theta);
        a2[i] = r*r;

        b0[i] = (1.0f-r)*(1.0f-r);

        y1_1[i]=0;
        y2_1[i]=0;
        y1_2[i]=0;
        y2_2[i]=0;
        env_state[i]=0;
    }
}

/* ================= PROCESS ================= */

void cochlea_process(float *frame,
                     int frame_size,
                     float *output)
{
    /* Reset output per frame */
    for(int ch=0; ch<COCHLEA_CHANNELS; ch++)
        output[ch] = 0.0f;

    for(int n=0;n<frame_size;n++)
    {
        float x = frame[n];

        for(int ch=0; ch<COCHLEA_CHANNELS; ch++)
        {
            /* First filter */
            float y1 = b0[ch]*x
                      -a1[ch]*y1_1[ch]
                      -a2[ch]*y2_1[ch];

            y2_1[ch]=y1_1[ch];
            y1_1[ch]=y1;

            /* Second filter */
            float y2 = b0[ch]*y1
                      -a1[ch]*y1_2[ch]
                      -a2[ch]*y2_2[ch];

            y2_2[ch]=y1_2[ch];
            y1_2[ch]=y2;

            /*  POWER instead of abs */
            float rect = y2 * y2;

            /* Faster envelope */
            env_state[ch] =
                ENV_DECAY * env_state[ch] +
                ENV_ALPHA * rect;

            /*  Keep PEAK energy (no averaging!) */
            if(env_state[ch] > output[ch])
                output[ch] = env_state[ch];
        }
    }

    /*  Prevent log(0) later */
    for(int ch=0; ch<COCHLEA_CHANNELS; ch++)
    {
        if(output[ch] < 1e-8f)
            output[ch] = 1e-8f;
    }
}
