#include <hls_math.h>

#include "lstm_top.h"
#include "lstm_weights.h"

static inline lstm_data_t sigmoid(lstm_data_t x) {
    float xf = (float)x;
    return (lstm_data_t)(1.0f / (1.0f + hls::expf(-xf)));
}

static inline lstm_data_t tanh_fixed(lstm_data_t x) {
    return (lstm_data_t)hls::tanhf((float)x);
}

template <int INPUT_SIZE>
static void lstm_step(
    const lstm_data_t x[INPUT_SIZE],
    const lstm_data_t h_prev[LSTM_HIDDEN_SIZE],
    const lstm_data_t c_prev[LSTM_HIDDEN_SIZE],
    const lstm_data_t w_ih[LSTM_GATE_SIZE][INPUT_SIZE],
    const lstm_data_t w_hh[LSTM_GATE_SIZE][LSTM_HIDDEN_SIZE],
    const lstm_data_t b_ih[LSTM_GATE_SIZE],
    const lstm_data_t b_hh[LSTM_GATE_SIZE],
    lstm_data_t h_next[LSTM_HIDDEN_SIZE],
    lstm_data_t c_next[LSTM_HIDDEN_SIZE]
) {
    lstm_data_t i_gate[LSTM_HIDDEN_SIZE];
    lstm_data_t f_gate[LSTM_HIDDEN_SIZE];
    lstm_data_t g_gate[LSTM_HIDDEN_SIZE];
    lstm_data_t o_gate[LSTM_HIDDEN_SIZE];

    Gate_Loop: for (int h = 0; h < LSTM_HIDDEN_SIZE; h++) {
        lstm_data_t sum_i = b_ih[h] + b_hh[h];
        lstm_data_t sum_f = b_ih[LSTM_HIDDEN_SIZE + h] + b_hh[LSTM_HIDDEN_SIZE + h];
        lstm_data_t sum_g = b_ih[(2 * LSTM_HIDDEN_SIZE) + h] + b_hh[(2 * LSTM_HIDDEN_SIZE) + h];
        lstm_data_t sum_o = b_ih[(3 * LSTM_HIDDEN_SIZE) + h] + b_hh[(3 * LSTM_HIDDEN_SIZE) + h];

        Input_Loop: for (int i = 0; i < INPUT_SIZE; i++) {
            lstm_data_t x_val = x[i];
            sum_i += x_val * w_ih[h][i];
            sum_f += x_val * w_ih[LSTM_HIDDEN_SIZE + h][i];
            sum_g += x_val * w_ih[(2 * LSTM_HIDDEN_SIZE) + h][i];
            sum_o += x_val * w_ih[(3 * LSTM_HIDDEN_SIZE) + h][i];
        }

        Hidden_Loop: for (int i = 0; i < LSTM_HIDDEN_SIZE; i++) {
            lstm_data_t h_val = h_prev[i];
            sum_i += h_val * w_hh[h][i];
            sum_f += h_val * w_hh[LSTM_HIDDEN_SIZE + h][i];
            sum_g += h_val * w_hh[(2 * LSTM_HIDDEN_SIZE) + h][i];
            sum_o += h_val * w_hh[(3 * LSTM_HIDDEN_SIZE) + h][i];
        }

        i_gate[h] = sigmoid(sum_i);
        f_gate[h] = sigmoid(sum_f);
        g_gate[h] = tanh_fixed(sum_g);
        o_gate[h] = sigmoid(sum_o);
    }

    State_Loop: for (int h = 0; h < LSTM_HIDDEN_SIZE; h++) {
        c_next[h] = (f_gate[h] * c_prev[h]) + (i_gate[h] * g_gate[h]);
        h_next[h] = o_gate[h] * tanh_fixed(c_next[h]);
    }
}

void lstm_top(
    lstm_data_t input[LSTM_IN_CH][LSTM_IN_LEN],
    lstm_data_t output[LSTM_NUM_CLASSES]
) {
    #pragma HLS INTERFACE m_axi port=input depth=LSTM_INPUT_SIZE bundle=gmem0
    #pragma HLS INTERFACE m_axi port=output depth=LSTM_NUM_CLASSES bundle=gmem0
    #pragma HLS INTERFACE s_axilite port=return
    #pragma HLS ALLOCATION operation instances=mul limit=LSTM_MUL_LIMIT
    #pragma HLS ALLOCATION operation instances=add limit=LSTM_ADD_LIMIT

    lstm_data_t local_in[LSTM_IN_CH][LSTM_IN_LEN];
    #pragma HLS BIND_STORAGE variable=local_in type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=lstm_weight_ih_l0 type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=lstm_weight_hh_l0 type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=lstm_weight_ih_l1 type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=lstm_weight_hh_l1 type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=lstm_fc_weight type=rom_2p impl=bram

    Copy_Input: for (int ch = 0; ch < LSTM_IN_CH; ch++) {
        for (int t = 0; t < LSTM_IN_LEN; t++) {
            #pragma HLS PIPELINE II=1
            local_in[ch][t] = input[ch][t];
        }
    }

    lstm_data_t layer1_h[LSTM_HIDDEN_SIZE] = {0};
    lstm_data_t layer1_c[LSTM_HIDDEN_SIZE] = {0};
    lstm_data_t layer2_h[LSTM_HIDDEN_SIZE] = {0};
    lstm_data_t layer2_c[LSTM_HIDDEN_SIZE] = {0};

    lstm_data_t x_t[LSTM_IN_CH];
    lstm_data_t layer1_next_h[LSTM_HIDDEN_SIZE];
    lstm_data_t layer1_next_c[LSTM_HIDDEN_SIZE];
    lstm_data_t layer2_next_h[LSTM_HIDDEN_SIZE];
    lstm_data_t layer2_next_c[LSTM_HIDDEN_SIZE];

    Time_Loop: for (int t = 0; t < LSTM_IN_LEN; t++) {
        Input_Vector: for (int ch = 0; ch < LSTM_IN_CH; ch++) {
            #pragma HLS PIPELINE II=1
            x_t[ch] = local_in[ch][t];
        }

        lstm_step<LSTM_IN_CH>(
            x_t,
            layer1_h,
            layer1_c,
            lstm_weight_ih_l0,
            lstm_weight_hh_l0,
            lstm_bias_ih_l0,
            lstm_bias_hh_l0,
            layer1_next_h,
            layer1_next_c
        );

        lstm_step<LSTM_HIDDEN_SIZE>(
            layer1_next_h,
            layer2_h,
            layer2_c,
            lstm_weight_ih_l1,
            lstm_weight_hh_l1,
            lstm_bias_ih_l1,
            lstm_bias_hh_l1,
            layer2_next_h,
            layer2_next_c
        );

        Update_Layer1: for (int h = 0; h < LSTM_HIDDEN_SIZE; h++) {
            layer1_h[h] = layer1_next_h[h];
            layer1_c[h] = layer1_next_c[h];
        }

        Update_Layer2: for (int h = 0; h < LSTM_HIDDEN_SIZE; h++) {
            layer2_h[h] = layer2_next_h[h];
            layer2_c[h] = layer2_next_c[h];
        }
    }

    FC: for (int cl = 0; cl < LSTM_NUM_CLASSES; cl++) {
        #pragma HLS PIPELINE II=1
        lstm_data_t sum = lstm_fc_bias[cl];
        for (int h = 0; h < LSTM_HIDDEN_SIZE; h++) {
            sum += layer2_h[h] * lstm_fc_weight[cl][h];
        }
        output[cl] = sum;
    }
}