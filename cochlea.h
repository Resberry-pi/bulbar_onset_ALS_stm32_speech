#ifndef COCHLEA_H
#define COCHLEA_H

#include <stdint.h>

#define COCHLEA_CHANNELS 24

void cochlea_init(void);

void cochlea_process(float *frame,
                     int frame_size,
                     float *output);

#endif
