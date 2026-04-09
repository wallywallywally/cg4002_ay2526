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
        {4.411,-0.720,0.934,0.927,-2.074,-0.280,-1.943,-2.563,-1.468},
        {-2.219,3.876,-1.365,-0.226,-1.120,-1.466,-2.309,-3.092,-2.161},
        {-1.891,-1.524,4.341,-0.150,5.260,-0.441,0.541,-0.377,0.044},
        {0.502,0.043,-1.639,2.850,-1.130,-2.860,-0.984,-2.946,-2.510},
        {-0.549,0.520,-2.638,1.079,0.988,-1.563,-2.013,-2.929,-2.372},
        {-3.006,0.545,-2.228,-1.983,0.769,1.564,-2.698,-2.373,0.213},
        {-2.580,0.654,-1.356,-0.893,-0.958,2.282,-2.951,-1.857,0.383},
        {-0.936,-1.066,-0.598,-1.007,-0.158,0.128,-0.652,4.618,-1.545},
        {0.100,-2.730,-0.679,0.945,-0.833,-2.071,-0.863,-3.361,0.979},
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