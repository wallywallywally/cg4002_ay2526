#ifndef CNN_TOP_H
#define CNN_TOP_H

#include "ap_fixed.h"

typedef ap_fixed<16, 6> data_t;

#define IN_CH 30                    // 30 features
#define IN_LEN 25                   // 25 rows - match window size
#define TOTAL_IN (IN_CH * IN_LEN)
#define NUM_CLASSES 10              // 10 gestures

// Layer-specific
#define KERNEL_SIZE 3
#define L1_OUT_CH 32
#define L1_WINDOW (IN_LEN - KERNEL_SIZE+ 1)
#define L2_OUT_CH 64
#define L2_WINDOW (L1_WINDOW - KERNEL_SIZE + 1)

void cnn_top(data_t input[IN_CH][IN_LEN], data_t output[NUM_CLASSES]);

#endif