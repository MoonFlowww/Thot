#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <unordered_map>

#include "utils/translators.h"

namespace Evaluations {
    inline void evaluate_binary(const std::vector<std::vector<float>>& predictions, const std::vector<std::vector<float>>& targets, const  std::vector<float>& latencies,
        size_t flops, size_t input_size, size_t output_size,bool verbose) {
        if (verbose) {
            std::cout << "\nBinary Classification Evaluation:\n";
            std::cout << "-------------------------------\n";
        }

        float accuracy = 0.0f;
        float precision = 0.0f;
        float recall = 0.0f;
        float f1 = 0.0f;

        for (size_t i = 0; i < predictions.size(); ++i) {
            if (verbose) {
                std::cout << "Input: [";
                for (float x : predictions[i]) std::cout << x << " ";
                std::cout << "] -> Output: [";
                for (float y : targets[i]) std::cout << y << " ";
                std::cout << "]\n" << std::endl;
            }

            float true_positives = 0.0f;
            float false_positives = 0.0f;
            float false_negatives = 0.0f;
            float true_negatives = 0.0f;

            for (size_t j = 0; j < predictions[i].size(); ++j) {
                float pred = predictions[i][j] >= 0.5f ? 1.0f : 0.0f;
                float target = targets[i][j];

                if (pred == 1.0f && target == 1.0f) true_positives += 1.0f;
                else if (pred == 1.0f && target == 0.0f) false_positives += 1.0f;
                else if (pred == 0.0f && target == 1.0f) false_negatives += 1.0f;
                else true_negatives += 1.0f;
            }

            accuracy += (true_positives + true_negatives) / predictions[i].size();
            precision += true_positives / (true_positives + false_positives + 1e-10f);
            recall += true_positives / (true_positives + false_negatives + 1e-10f);
        }

        accuracy /= predictions.size();
        precision /= predictions.size();
        recall /= predictions.size();
        f1 = 2.0f * (precision * recall) / (precision + recall + 1e-10f);

        float total_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0f);
        float avg_latency = total_latency / latencies.size();
        float sq_sum = std::inner_product(latencies.begin(), latencies.end(), latencies.begin(), 0.0f);
        float std_latency = std::sqrt(sq_sum / latencies.size() - avg_latency * avg_latency);
        float skew_latency = 0.0f;
        for (float l : latencies) skew_latency += std::pow(l - avg_latency, 3);
        skew_latency /= latencies.size();
        skew_latency /= std::pow(std_latency, 3) + 1e-10f;
        std::unordered_map<int,int> freq;
        for (float l : latencies) freq[static_cast<int>(std::round(l))]++;
        float mode_latency = latencies.empty() ? 0.0f : std::round(latencies[0]);
        int max_count = 0;
        for (auto& kv : freq) {
            if (kv.second > max_count) { max_count = kv.second; mode_latency = kv.first; }
        }
        size_t model_input_bytes = predictions.size() * input_size * sizeof(float);
        size_t model_output_bytes = predictions.size() * output_size * sizeof(float);
        float total_seconds = total_latency / 1000.0f;
        float input_bps  = total_seconds > 0 ? static_cast<float>(model_input_bytes) / total_seconds : 0.0f;
        float output_bps = total_seconds > 0 ? static_cast<float>(model_output_bytes) / total_seconds : 0.0f;
        float throughput = 1.0f / avg_latency;


        if (verbose) {
            std::cout << "\n *~~~~~~~~~ Metrics ~~~~~~~~~*\n";
            std::cout << " | Accuracy: " << accuracy * 100.0f << "%\n";
            std::cout << " | Precision: " << precision * 100.0f << "%\n";
            std::cout << " | Recall: " << recall * 100.0f << "%\n";
            std::cout << " | F1 Score: " << f1 * 100.0f << "%\n";
            std::cout << " *~~~~~~~~~~~~~~~~~~~~~~~~~~~~*" << std::endl;
            std::cout << " | Latency Average : " << Thot::format_time(avg_latency) << "\n";
            std::cout << " | Latency Std Dev: " << Thot::format_time(std_latency) << "\n";
            std::cout << " | Latency Skew: " << skew_latency << "\n";
            std::cout << " | Latency Mode: " << Thot::format_time(mode_latency) << "\n";
            std::cout << " *~~~~~~~~~~~~~~~~~~~~~~~~~~~~*" << std::endl;

            std::cout << " | Input Bytes/s: " << Thot::formatBytes(input_bps) << "\n";
            std::cout << " | Output Bytes/s: " << Thot::formatBytes(output_bps) << "\n";
            std::cout << " | Throughput: " << throughput << " FLOPS\n";
            std::cout << " *~~~~~~~~~~~~~~~~~~~~~~~~~~~~*" << std::endl;

        }
    }
}