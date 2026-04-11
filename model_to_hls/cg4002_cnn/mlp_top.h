#ifndef MLP_TOP_H
#define MLP_TOP_H

#include "ap_fixed.h"

typedef ap_fixed<32, 12> mlp_data_t;

#define MLP_IN_CH 30
#define MLP_IN_LEN 25
#define MLP_INPUT_SIZE (MLP_IN_CH * MLP_IN_LEN)
#define MLP_H1_SIZE 128
#define MLP_H2_SIZE 64
#define MLP_H3_SIZE 32
#define MLP_NUM_CLASSES 9

void mlp_top(mlp_data_t input[MLP_IN_CH][MLP_IN_LEN], mlp_data_t output[MLP_NUM_CLASSES]);

#endif