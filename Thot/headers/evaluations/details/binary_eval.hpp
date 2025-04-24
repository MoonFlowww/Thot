#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace Evaluations {
    inline void evaluate_binary(const std::vector<std::vector<float>>& predictions, const std::vector<std::vector<float>>& targets, const  std::vector<float>& latencies, size_t flops, bool verbose) {
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

        float avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / latencies.size();
        float throughput = 1.0f / avg_latency;

        if (verbose) {
            std::cout << "\nMetrics:\n";
            std::cout << "Accuracy: " << accuracy * 100.0f << "%\n";
            std::cout << "Precision: " << precision * 100.0f << "%\n";
            std::cout << "Recall: " << recall * 100.0f << "%\n";
            std::cout << "F1 Score: " << f1 * 100.0f << "%\n";
            std::cout << "Average Latency: " << avg_latency << " ms\n";
            std::cout << "Throughput: " << throughput << " FLOPS\n";
        }
    }
}