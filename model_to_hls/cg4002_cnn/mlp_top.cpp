#include "mlp_top.h"
#include "mlp_weights.h"

static inline mlp_data_t relu(mlp_data_t x) {
    return (x > 0) ? x : (mlp_data_t)0;
}

void mlp_top(
    mlp_data_t input[MLP_IN_CH][MLP_IN_LEN],
    mlp_data_t output[MLP_NUM_CLASSES]
) {
    #pragma HLS INTERFACE m_axi port=input depth=MLP_INPUT_SIZE bundle=gmem0
    #pragma HLS INTERFACE m_axi port=output depth=MLP_NUM_CLASSES bundle=gmem0
    #pragma HLS INTERFACE s_axilite port=return
    #pragma HLS ALLOCATION operation instances=mul limit=MLP_MUL_LIMIT
    #pragma HLS ALLOCATION operation instances=add limit=MLP_ADD_LIMIT

    mlp_data_t flat_input[MLP_INPUT_SIZE];
    mlp_data_t layer1[MLP_H1_SIZE];
    mlp_data_t layer2[MLP_H2_SIZE];
    mlp_data_t layer3[MLP_H3_SIZE];

    #pragma HLS BIND_STORAGE variable=flat_input type=ram_2p impl=bram

    Flatten_Input: for (int ch = 0; ch < MLP_IN_CH; ch++) {
        for (int t = 0; t < MLP_IN_LEN; t++) {
            #pragma HLS PIPELINE II=1
            flat_input[(ch * MLP_IN_LEN) + t] = input[ch][t];
        }
    }

    FC1: for (int out_idx = 0; out_idx < MLP_H1_SIZE; out_idx++) {
        #pragma HLS PIPELINE II=1
        mlp_data_t sum = mlp_fc1_bias[out_idx];
        for (int in_idx = 0; in_idx < MLP_INPUT_SIZE; in_idx++) {
            #pragma HLS LOOP_TRIPCOUNT min=750 max=750
            sum += flat_input[in_idx] * mlp_fc1_weight[out_idx][in_idx];
        }
        layer1[out_idx] = relu(sum);
    }

    FC2: for (int out_idx = 0; out_idx < MLP_H2_SIZE; out_idx++) {
        #pragma HLS PIPELINE II=1
        mlp_data_t sum = mlp_fc2_bias[out_idx];
        for (int in_idx = 0; in_idx < MLP_H1_SIZE; in_idx++) {
            #pragma HLS LOOP_TRIPCOUNT min=128 max=128
            sum += layer1[in_idx] * mlp_fc2_weight[out_idx][in_idx];
        }
        layer2[out_idx] = relu(sum);
    }

    FC3: for (int out_idx = 0; out_idx < MLP_H3_SIZE; out_idx++) {
        #pragma HLS PIPELINE II=1
        mlp_data_t sum = mlp_fc3_bias[out_idx];
        for (int in_idx = 0; in_idx < MLP_H2_SIZE; in_idx++) {
            #pragma HLS LOOP_TRIPCOUNT min=64 max=64
            sum += layer2[in_idx] * mlp_fc3_weight[out_idx][in_idx];
        }
        layer3[out_idx] = relu(sum);
    }

    FC4: for (int out_idx = 0; out_idx < MLP_NUM_CLASSES; out_idx++) {
        #pragma HLS PIPELINE II=1
        mlp_data_t sum = mlp_fc4_bias[out_idx];
        for (int in_idx = 0; in_idx < MLP_H3_SIZE; in_idx++) {
            #pragma HLS LOOP_TRIPCOUNT min=32 max=32
            sum += layer3[in_idx] * mlp_fc4_weight[out_idx][in_idx];
        }
        output[out_idx] = sum;
    }
}