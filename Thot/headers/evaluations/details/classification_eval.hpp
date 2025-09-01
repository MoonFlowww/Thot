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
    inline void evaluate_classification(const std::vector<std::vector<float>>& predictions, const std::vector<std::vector<float>>& targets, const std::vector<float>& latencies,
        size_t flops, size_t input_size, size_t output_size,bool verbose) {
        if (verbose) {
            std::cout << "\nMulti-class Classification Evaluation:\n";
            std::cout << "------------------------------------\n";
        }

        size_t num_classes = predictions[0].size();
        std::vector<float> class_precision(num_classes, 0.0f);
        std::vector<float> class_recall(num_classes, 0.0f);
        std::vector<float> class_f1(num_classes, 0.0f);

        std::vector<int> true_positives(num_classes, 0);
        std::vector<int> false_positives(num_classes, 0);
        std::vector<int> false_negatives(num_classes, 0);
        std::vector<int> support(num_classes, 0); // count of samples per class

        int correct = 0;

        for (size_t i = 0; i < predictions.size(); ++i) {
            size_t pred_class = std::distance(predictions[i].begin(),
                std::max_element(predictions[i].begin(), predictions[i].end()));
            size_t actual_class = std::distance(targets[i].begin(),
                std::max_element(targets[i].begin(), targets[i].end()));

            support[actual_class]++;

            if (pred_class == actual_class) {
                correct++;
                true_positives[actual_class]++;
            }
            else {
                false_positives[pred_class]++;
                false_negatives[actual_class]++;
            }
        }

        // Compute per-class metrics
        for (size_t c = 0; c < num_classes; ++c) {
            float prec = true_positives[c] / float(true_positives[c] + false_positives[c] + 1e-10f);
            float rec = true_positives[c] / float(true_positives[c] + false_negatives[c] + 1e-10f);
            float f1 = 2.0f * prec * rec / (prec + rec + 1e-10f);

            class_precision[c] = prec;
            class_recall[c] = rec;
            class_f1[c] = f1;
        }

        // Global (micro-averaged) metrics
        int total_tp = std::accumulate(true_positives.begin(), true_positives.end(), 0);
        int total_fp = std::accumulate(false_positives.begin(), false_positives.end(), 0);
        int total_fn = std::accumulate(false_negatives.begin(), false_negatives.end(), 0);

        float global_precision = total_tp / float(total_tp + total_fp + 1e-10f);
        float global_recall = total_tp / float(total_tp + total_fn + 1e-10f);
        float global_f1 = 2.0f * global_precision * global_recall / (global_precision + global_recall + 1e-10f);

        float accuracy = float(correct) / predictions.size();
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

        //TODO: remove 28*28 via size input
        size_t model_input_bytes = predictions.size() * 28 * 28 * sizeof(float);
        size_t model_output_bytes = predictions.size() * num_classes * sizeof(float);

        float total_seconds = total_latency / 1000.0f;
        float input_bps  = total_seconds > 0 ? static_cast<float>(model_input_bytes) / total_seconds : 0.0f;
        float output_bps = total_seconds > 0 ? static_cast<float>(model_output_bytes) / total_seconds : 0.0f;

        float throughput = 1.0f / avg_latency;

        if (verbose) {
            std::cout << "\n *~~~~~~~~~ Metrics ~~~~~~~~~*\n";
            std::cout << " | Overall Accuracy: " << accuracy * 100.0f << "%\n";
            std::cout << " | Global Precision: " << global_precision * 100.0f << "%\n";
            std::cout << " | Global Recall:    " << global_recall * 100.0f << "%\n";
            std::cout << " | Global F1 Score:  " << global_f1 * 100.0f << "%\n";
            std::cout << " *~~~~~~~~~~~~~~~~~~~~~~~~~~~~*" << std::endl;


            std::cout << "\n *~~~~ Per-class Metrics ~~~~*\n";
            for (size_t c = 0; c < num_classes; ++c) {
                std::cout << " | Class " << c << ":\n";
                std::cout << " |   Precision: " << class_precision[c] * 100.0f << "%\n";
                std::cout << " |   Recall: " << class_recall[c] * 100.0f << "%\n";
                std::cout << " |   F1 Score: " << class_f1[c] * 100.0f << "%\n";
            }

            std::cout << " *~~~~~~~~~~~~~~~~~~~~~~~~~~~~*" << std::endl;
            std::cout << " | Latency Average : " << avg_latency << " ms\n";
            std::cout << " | Latency Std Dev: " << std_latency << " ms\n";
            std::cout << " | Latency Skew: " << skew_latency << "\n";
            std::cout << " | Latency Mode: " << mode_latency << " ms\n";
            std::cout << " *~~~~~~~~~~~~~~~~~~~~~~~~~~~~*" << std::endl;

            std::cout << " | Input Bytes/s: " << Thot::formatBytes(input_bps) << "\n";
            std::cout << " | Output Bytes/s: " << Thot::formatBytes(output_bps) << "\n";
            std::cout << " | Throughput: " << throughput << " FLOPS\n";
            std::cout << " *~~~~~~~~~~~~~~~~~~~~~~~~~~~~*" << std::endl;
        }
    }
}
