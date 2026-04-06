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
        {5.136,1.782,3.043,2.574,-0.199,-2.359,-6.073,-2.157,-2.559},
        {-0.696,4.987,-1.447,3.836,-1.081,-5.085,1.935,-3.099,-1.132},
        {3.299,-0.518,5.827,-3.797,5.726,-3.453,1.080,-4.404,-8.156},
        {1.015,2.414,1.075,4.670,0.845,-6.354,-4.881,1.129,-1.547},
        {1.228,0.350,0.445,1.542,2.624,-1.533,-2.191,-1.588,-1.971},
        {-0.868,-1.618,-1.441,-1.372,2.254,2.799,1.997,-3.473,0.628},
        {-0.710,-0.642,-2.015,-0.830,1.286,2.698,3.003,-3.691,-0.220},
        {3.299,18.172,-0.060,-26.926,5.437,-37.675,-16.617,21.361,16.890},
        {1.610,-0.604,2.408,0.901,1.619,-2.068,-6.908,0.067,1.512},
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