#include <cmath>
#include <iostream>

#include "lstm_top.h"

static void fill_test_input(lstm_data_t input[LSTM_IN_CH][LSTM_IN_LEN], int sample_idx) {
    for (int ch = 0; ch < LSTM_IN_CH; ch++) {
        for (int t = 0; t < LSTM_IN_LEN; t++) {
            int pattern = ((ch + 1) * (t + 1) + sample_idx) % 11;
            input[ch][t] = (lstm_data_t)((pattern - 5) * 0.1f);
        }
    }
}

static bool output_is_finite(const lstm_data_t output[LSTM_NUM_CLASSES]) {
    for (int i = 0; i < LSTM_NUM_CLASSES; i++) {
        float value = (float)output[i];
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

int main() {
    const int num_tests = 3;

    for (int sample_idx = 0; sample_idx < num_tests; sample_idx++) {
        lstm_data_t test_input[LSTM_IN_CH][LSTM_IN_LEN];
        lstm_data_t output[LSTM_NUM_CLASSES];

        fill_test_input(test_input, sample_idx);
        lstm_top(test_input, output);

        std::cout << "[TEST] sample_" << sample_idx << " outputs: ";
        for (int i = 0; i < LSTM_NUM_CLASSES; i++) {
            std::cout << (float)output[i];
            if (i + 1 < LSTM_NUM_CLASSES) {
                std::cout << ", ";
            }
        }
        std::cout << std::endl;

        if (!output_is_finite(output)) {
            std::cout << "[FAIL] Non-finite output detected in sample_" << sample_idx << std::endl;
            return 1;
        }
    }

    std::cout << "[PASS] LSTM C/RTL cosim testbench completed successfully" << std::endl;
    return 0;
}