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
        {3.155,-1.760,-1.031,0.313,-2.594,-3.771,-4.470,-2.919,-3.106},
        {-0.184,1.679,-1.908,3.074,-1.888,-1.806,0.201,-2.019,-1.178},
        {-2.216,-1.486,2.047,-2.426,3.038,-2.822,-2.088,-2.789,-2.878},
        {0.568,0.737,-1.413,3.192,-1.552,-1.424,-1.435,0.411,-1.603},
        {-0.585,-0.310,-1.393,0.448,1.547,-0.837,-1.844,-1.367,-2.251},
        {-1.000,-1.826,-2.837,-2.037,0.724,-0.004,1.496,-2.916,0.249},
        {-1.266,-1.636,-2.210,-1.221,-0.781,0.028,1.629,-2.703,0.125},
        {-1.426,1.954,-0.161,0.895,0.610,-1.624,1.252,5.372,0.327},
        {2.180,-2.421,-0.842,1.295,-0.382,-1.108,-3.314,-3.211,0.668},
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