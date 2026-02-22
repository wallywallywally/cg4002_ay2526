#ifndef CNN_TOP_H
#define CNN_TOP_H

#include "ap_fixed.h"

typedef ap_fixed<16, 6> data_t;

#define IN_CH 8                     // 8 features
#define IN_LEN 25                   // 25 rows
#define NUM_CLASSES 8               // 8 gestures

// Layer-specific
#define L1_OUT_CH 32
#define L2_OUT_CH 64

void cnn_top(data_t input[IN_CH][IN_LEN], data_t output[NUM_CLASSES]);

#endif