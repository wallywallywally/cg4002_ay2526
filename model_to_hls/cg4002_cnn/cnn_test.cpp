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
        "sample_knead_1.txt",
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
        {22.104,-111.683,-25.098,-14.132,-56.815,-32.793,-120.213,0.526,5.921,11.574},
        {-15.153,-2.517,-7.941,7.467,-128.308,-79.320,-15.829,12.094,12.351,-0.699},
        {-36.941,-209.598,5.532,-48.764,-26.334,-48.481,-2.276,-12.300,-25.541,-13.817},
        {0.694,-9.235,-10.269,10.343,-99.605,-57.681,-16.113,6.644,-0.887,3.094},
        {-44.807,-173.790,-41.219,-26.490,9.675,-38.824,-73.555,-4.714,-11.299,-2.294},
        {-9.730,4.066,-7.151,4.847,-124.151,-84.661,-19.071,9.033,5.518,11.174},
        {-57.692,-45.577,9.274,-14.599,-59.305,-114.236,10.861,-0.476,-20.066,-10.063},
        {-109.913,4.999,-14.609,-4.680,-118.403,-193.765,-12.085,8.941,-15.326,-11.053},
        {-9.132,-2.663,-16.227,6.536,-133.770,-65.250,-27.794,14.099,18.073,9.977},
        {0.153,-45.033,-16.057,-8.320,-30.849,-77.444,-99.431,-0.622,1.149,13.279}
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