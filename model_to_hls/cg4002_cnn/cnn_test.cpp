#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include "cnn_top.h"

struct Pred {
    int id;
    float score;
};

bool comparePreds(Pred a, Pred b) {
    return a.score > b.score;
}

int main() {
    std::vector<std::string> test_files = {
        "sample_chop_1.txt",
        "sample_select_1.txt",
        "sample_shake_1.txt",
        "sample_squeeze_1.txt",
        "sample_stir_1.txt",
        "sample_swipe_l_1.txt",
        "sample_swipe_r_1.txt",
        "sample_twist_l_1.txt",
        "sample_twist_r_1.txt"
    };

    // Match above order
    float actual[][NUM_CLASSES] = {
        {5.002,-0.336,0.632,-0.086,0.765,-0.505,-1.661,-3.587,-0.489},
        {-1.870,3.580,-0.771,0.875,0.808,-0.052,-0.913,-2.338,0.098},
        {0.863,-1.176,2.064,-2.051,4.140,-1.103,-0.556,-1.480,-1.224},
        {0.176,0.584,0.922,2.512,0.582,-1.352,-0.933,-2.876,-0.198},
        {0.227,1.161,0.027,1.407,1.453,-0.052,-1.472,-2.694,-0.572},
        {-1.364,0.977,-0.778,0.214,0.908,1.746,-1.776,-2.147,1.711},
        {-1.419,1.205,-1.231,0.445,0.549,2.219,-1.777,-1.808,1.272},
        {-0.306,1.072,-1.664,-0.091,-1.559,-0.642,-0.188,3.632,-0.152},
        {1.346,-0.191,0.446,1.574,1.110,-0.596,-1.471,-2.969,0.383},
    };

    std::vector<std::string> failed_files;

    for (size_t f = 0; f < test_files.size(); f++) {
        data_t test_input[IN_CH][IN_LEN];
        data_t output[NUM_CLASSES];

        std::ifstream infile(test_files[f]);
        for(int i=0; i<IN_CH; i++) {
            for(int j=0; j<IN_LEN; j++) {
                float val;
                infile >> val;
                test_input[i][j] = (data_t)val;
            }
        }
        infile.close();

        cnn_top(test_input, output);

        std::vector<Pred> hls_rank, py_rank;
        for (int i = 0; i < NUM_CLASSES; i++) {
            hls_rank.push_back({i, (float)output[i]});
            py_rank.push_back({i, actual[f][i]});
        }

        std::sort(hls_rank.begin(), hls_rank.end(), comparePreds);
        std::sort(py_rank.begin(), py_rank.end(), comparePreds);

        int K = 3;
        bool topk_match = true;
        for (int i = 0; i < K; i++) {
            if (hls_rank[i].id != py_rank[i].id) {
                topk_match = false;
                break;
            }
        }

        if (topk_match) {
            std::cout << "[PASS] " << test_files[f] << " (Top-" << K << " Match)" << std::endl;
        } else {
            std::cout << "[FAIL] " << test_files[f] << " (Top-" << K << " Mismatch! HLS Winner: " 
                      << hls_rank[0].id << ", PY Winner: " << py_rank[0].id << ")" << std::endl;
            
            // Print HLS Ranking
            std::cout << "  HLS Top-K: ";
            for (int i = 0; i < K; i++) {
                std::cout << "ID:" << hls_rank[i].id << "(" << (float)hls_rank[i].score << ") ";
            }
            std::cout << std::endl;

            // Print Python Ranking
            std::cout << "  PY  Top-K: ";
            for (int i = 0; i < K; i++) {
                std::cout << "ID:" << py_rank[i].id << "(" << (float)py_rank[i].score << ") ";
            }
            std::cout << std::endl;

            failed_files.push_back(test_files[f]);
        }
    }

    if (!failed_files.empty()) {
        std::cout << std::endl << "FAILED FILES:" << std::endl;
        for (const auto& name : failed_files) {
            std::cout << "  - " << name << std::endl;
        }
        std::cout << "---------------------------------------------" << std::endl;
        return 1;
    }

    return 0;
}