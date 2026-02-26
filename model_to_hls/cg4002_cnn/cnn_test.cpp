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
    // TOOD: load a sample for each gesture class
    std::vector<std::string> test_files = {
        "sample_select_1.txt", 
        // "sample_back_1.txt", 
        // "sample_noise.txt"
    };

    // Match above order
    float actual[][NUM_CLASSES] = {
        {-52.494,-53.161,4.739,-52.302,-52.714,-52.719,-2.312,-52.940,-54.809,-53.036},
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