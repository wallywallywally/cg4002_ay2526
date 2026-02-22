#include <iostream>
#include "cnn_top.h"

int main() {
    // Dummy input
    // TOOD: test with real data
    data_t test_input[IN_CH][IN_LEN];
    data_t hardware_output[NUM_CLASSES];

    for(int i=0; i<IN_CH; i++)
        for(int j=0; j<IN_LEN; j++)
            test_input[i][j] = 0.5;

    // Test
    std::cout << "Starting hardware simulation..." << std::endl;
    cnn_top(test_input, hardware_output);
    std::cout << "Simulation finished!" << std::endl;

    data_t max_val = hardware_output[0];
    int prediction = 0;

    for(int i=0; i<NUM_CLASSES; i++) {
        std::cout << "Class " << i << " score: " << hardware_output[i] << std::endl;
        if(hardware_output[i] > max_val) {
            max_val = hardware_output[i];
            prediction = i;
        }
    }
    std::cout << "-----------------------------" << std::endl;
    std::cout << "Max Score: " << (float)max_val << std::endl;
    std::cout << "Final Prediction (Class Index): " << prediction << std::endl;
    std::cout << "-----------------------------" << std::endl;

    return 0;
}