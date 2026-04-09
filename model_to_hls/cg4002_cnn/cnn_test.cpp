#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include "cnn_top.h"

struct Pred {
    int id;
    float score;
};

bool comparePreds(Pred a, Pred b) {
    return a.score > b.score;
}

int topKOverlap(const std::vector<Pred>& a, const std::vector<Pred>& b, int k) {
    std::unordered_set<int> ids;
    for (int i = 0; i < k; i++) {
        ids.insert(a[i].id);
    }

    int overlap = 0;
    for (int i = 0; i < k; i++) {
        if (ids.find(b[i].id) != ids.end()) {
            overlap++;
        }
    }
    return overlap;
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
        {-0.179,-6.369,-2.402,-4.325,-3.797,-4.423,-3.974,-5.640,-3.856},
        {-2.920,4.150,-1.604,-0.014,-4.435,-3.228,-1.253,-3.870,-3.212},
        {-6.054,-4.288,1.511,-3.832,1.860,-2.661,-3.130,-3.177,-3.354},
        {-1.908,-0.581,-2.808,1.454,-2.979,-4.931,-2.893,-3.567,-4.763},
        {-1.736,-1.969,-3.741,-0.974,0.581,-3.571,-3.792,-3.753,-5.147},
        {-3.007,1.257,-2.573,-2.552,-0.197,2.422,-2.777,-3.425,-1.226},
        {-2.772,2.078,-2.865,0.576,-2.686,3.153,-1.348,-2.305,-1.100},
        {0.933,0.586,-0.276,0.500,1.781,-0.397,1.760,6.302,-0.786},
        {-1.067,-1.848,-3.185,0.040,-2.671,-2.651,-4.812,-3.908,-3.424},
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
        int overlap = topKOverlap(hls_rank, py_rank, K);
        bool top1_match = (hls_rank[0].id == py_rank[0].id);
        bool topk_match = (overlap >= 2);

        if (topk_match) {
            std::cout << "[PASS] " << test_files[f]
                      << " (Top-" << K << " overlap=" << overlap
                      << ", Top-1 " << (top1_match ? "match" : "mismatch") << ")"
                      << std::endl;
        } else {
            std::cout << "[FAIL] " << test_files[f]
                      << " (Top-" << K << " overlap=" << overlap
                      << ", HLS Winner: " << hls_rank[0].id
                      << ", PY Winner: " << py_rank[0].id << ")"
                      << std::endl;
            
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