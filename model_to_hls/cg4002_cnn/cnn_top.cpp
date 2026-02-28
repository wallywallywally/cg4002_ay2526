#include <hls_math.h>
#include "cnn_top.h"
#include "weights.h"

void cnn_top(
    data_t input[IN_CH][IN_LEN], 
    data_t output[NUM_CLASSES]
) {
    // AXI ports
    #pragma HLS INTERFACE m_axi port=input depth=TOTAL_IN bundle=gmem0
    #pragma HLS INTERFACE m_axi port=output depth=NUM_CLASSES bundle=gmem0
    #pragma HLS INTERFACE s_axilite port=return

    // Copy to local memory
    data_t local_in[IN_CH][IN_LEN];
    // #pragma HLS ARRAY_RESHAPE variable=local_in cyclic factor=2 dim=1
    #pragma HLS BIND_STORAGE variable=local_in type=ram_2p impl=bram

    for(int ic=0; ic<IN_CH; ic++) {
        for(int j=0; j<IN_LEN; j++) {
            #pragma HLS PIPELINE II=1
            local_in[ic][j] = input[ic][j];
        }
    }
    
    // Internal Buffers for Layer Outputs
    static data_t layer1_out[L1_OUT_CH][L1_WINDOW];
    // #pragma HLS ARRAY_RESHAPE variable=layer1_out cyclic factor=2 dim=1
    static data_t layer2_out[L2_OUT_CH][L2_WINDOW];
    // #pragma HLS ARRAY_RESHAPE variable=layer2_out cyclic factor=2 dim=1
    static data_t pool_out[L2_OUT_CH];

    #pragma HLS BIND_STORAGE variable=conv_block_0_weight type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=conv_block_3_weight type=rom_2p impl=bram


    // --- LAYER 1: Conv1d(8, 32, k=3) ---
    L1_Conv: for(int oc=0; oc < L1_OUT_CH; oc++) {
        for(int i=0; i < L1_WINDOW; i++) {
            #pragma HLS PIPELINE II=8
            #pragma HLS ALLOCATION operation instances=mul limit=4
            #pragma BIND_OP variable=sum op=add impl=dsp
            data_t sum = conv_block_0_bias[oc];
            for(int ic=0; ic < IN_CH; ic++) {
                for(int k=0; k < KERNEL_SIZE_1; k++) {
                    int idx = (oc * IN_CH * KERNEL_SIZE_1) + (ic * KERNEL_SIZE_1) + k;
                    sum += local_in[ic][i+k] * conv_block_0_weight[idx];
                }
            }
            
            // LeakyReLU
            data_t leaky;
            leaky = (sum >> 4) + (sum >> 5);
            layer1_out[oc][i] = (sum > 0) ? sum : leaky;
        }
    }

    // --- LAYER 2: Conv1d(32, 64, k=3) ---
    L2_Conv: for(int oc=0; oc < L2_OUT_CH; oc++) {
        for(int i=0; i < L2_WINDOW; i++) {
            #pragma HLS PIPELINE II=8
            #pragma HLS ALLOCATION operation instances=mul limit=8
            #pragma BIND_OP variable=sum op=add impl=dsp
            data_t sum = conv_block_3_bias[oc];
            for(int ic=0; ic < L1_OUT_CH; ic++) {
                for(int k=0; k < KERNEL_SIZE_2; k++) {
                    int idx = (oc * L1_OUT_CH * KERNEL_SIZE_2) + (ic * KERNEL_SIZE_2) + k;
                    sum += layer1_out[ic][i+k] * conv_block_3_weight[idx];
                }
            }
            
            data_t leaky;
            leaky = (sum >> 4) + (sum >> 5);
            layer2_out[oc][i] = (sum > 0) ? sum : leaky;
        }
    }

    // --- LAYER 3: AdaptiveAvgPool1d(1) ---
    Pool: for(int c=0; c < L2_OUT_CH; c++) {
        #pragma HLS PIPELINE II=2
        data_t acc = 0;
        for(int i=0; i < L2_WINDOW; i++) {
            acc += layer2_out[c][i];
        }
        pool_out[c] = acc / (data_t)L2_WINDOW;
    }

    // --- LAYER 4: Fully Connected (64 -> 8) ---
    FC: for(int cl=0; cl < NUM_CLASSES; cl++) {
        #pragma HLS PIPELINE II=2
        data_t sum = fc_bias[cl];
        for(int i=0; i < L2_OUT_CH; i++) {
            sum += pool_out[i] * fc_weight[(cl * L2_OUT_CH) + i];
        }
        output[cl] = sum;
    }
}