#include <hls_math.h>
#include "cnn_top.h"
#include "weights.h"

void cnn_top(
    data_t input[IN_CH][IN_LEN], 
    data_t output[NUM_CLASSES]
) {
    // AXI ports
    #pragma HLS INTERFACE m_axi port=input depth=200 bundle=gmem0
    #pragma HLS INTERFACE m_axi port=output depth=8 bundle=gmem0
    #pragma HLS INTERFACE s_axilite port=return

    // Copy to local memory
    data_t local_in[IN_CH][IN_LEN];
    #pragma HLS ARRAY_PARTITION variable=local_in complete dim=1

    for(int ic=0; ic<IN_CH; ic++) {
        for(int j=0; j<IN_LEN; j++) {
            #pragma HLS PIPELINE
            local_in[ic][j] = input[ic][j];
        }
    }
    
    // Internal Buffers for Layer Outputs
    static data_t layer1_out[L1_OUT_CH][23];        // L1: 25 - 3 + 1 = 23
    static data_t layer2_out[L2_OUT_CH][21];        // L2: 23 - 3 + 1 = 21
    static data_t pool_out[L2_OUT_CH];

    // --- LAYER 1: Conv1d(8, 32, k=3) ---
    L1_Conv: for(int oc=0; oc < L1_OUT_CH; oc++) {
        for(int i=0; i < 23; i++) {
            #pragma HLS PIPELINE II=1
            data_t sum = conv_block_0_bias[oc];
            for(int ic=0; ic < IN_CH; ic++) {
                for(int k=0; k < 3; k++) {
                    int idx = (oc * IN_CH * 3) + (ic * 3) + k;
                    sum += input[ic][i+k] * conv_block_0_weight[idx];
                }
            }
            layer1_out[oc][i] = (sum > 0) ? sum : (data_t)0;
        }
    }

    // --- LAYER 2: Conv1d(32, 64, k=3) ---
    L2_Conv: for(int oc=0; oc < L2_OUT_CH; oc++) {
        for(int i=0; i < 21; i++) {
            #pragma HLS PIPELINE II=1
            data_t sum = conv_block_2_bias[oc];
            for(int ic=0; ic < L1_OUT_CH; ic++) {
                for(int k=0; k < 3; k++) {
                    int idx = (oc * L1_OUT_CH * 3) + (ic * 3) + k;
                    sum += layer1_out[ic][i+k] * conv_block_2_weight[idx];
                }
            }
            layer2_out[oc][i] = (sum > 0) ? sum : (data_t)0;
        }
    }

    // --- LAYER 3: AdaptiveAvgPool1d(1) ---
    Pool: for(int c=0; c < L2_OUT_CH; c++) {
        #pragma HLS PIPELINE
        data_t acc = 0;
        for(int i=0; i < 21; i++) {
            acc += layer2_out[c][i];
        }
        pool_out[c] = acc / 21; // Average over the new length of 21
    }

    // --- LAYER 4: Fully Connected (64 -> 8) ---
    FC: for(int cl=0; cl < NUM_CLASSES; cl++) {
        #pragma HLS PIPELINE
        data_t sum = fc_bias[cl];
        for(int i=0; i < L2_OUT_CH; i++) {
            sum += pool_out[i] * fc_weight[(cl * L2_OUT_CH) + i];
        }
        output[cl] = sum;
    }
}