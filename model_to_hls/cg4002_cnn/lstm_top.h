#ifndef LSTM_TOP_H
#define LSTM_TOP_H

#include "ap_fixed.h"

typedef ap_fixed<32, 12> lstm_data_t;

#define LSTM_IN_CH 30
#define LSTM_IN_LEN 25
#define LSTM_HIDDEN_SIZE 64
#define LSTM_NUM_LAYERS 2
#define LSTM_GATE_SIZE (4 * LSTM_HIDDEN_SIZE)
#define LSTM_NUM_CLASSES 9
#define LSTM_INPUT_SIZE (LSTM_IN_CH * LSTM_IN_LEN)

// HLS resource caps
#define LSTM_MUL_LIMIT 2
#define LSTM_ADD_LIMIT 4

void lstm_top(lstm_data_t input[LSTM_IN_CH][LSTM_IN_LEN], lstm_data_t output[LSTM_NUM_CLASSES]);

#endif