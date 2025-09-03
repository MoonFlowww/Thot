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

        // Macro-averaged metrics
        float macro_precision = std::accumulate(class_precision.begin(), class_precision.end(), 0.0f) / num_classes;
        float macro_recall    = std::accumulate(class_recall.begin(), class_recall.end(), 0.0f) / num_classes;
        float macro_f1        = std::accumulate(class_f1.begin(), class_f1.end(), 0.0f) / num_classes;

        // Weighted (support-based) metrics
        int total_support = std::accumulate(support.begin(), support.end(), 0);
        float weighted_precision = 0.0f;
        float weighted_recall    = 0.0f;
        float weighted_f1        = 0.0f;
        if (total_support > 0) {
            for (size_t c = 0; c < num_classes; ++c) {
                weighted_precision += class_precision[c] * support[c];
                weighted_recall    += class_recall[c] * support[c];
                weighted_f1        += class_f1[c] * support[c];
            }
            weighted_precision /= total_support;
            weighted_recall    /= total_support;
            weighted_f1        /= total_support;
        }


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

        size_t model_input_bytes = predictions.size() * input_size * sizeof(float);
        size_t model_output_bytes = predictions.size() * num_classes * sizeof(float);

        float total_seconds = total_latency / 1000.0f;
        float input_bps  = total_seconds > 0 ? static_cast<float>(model_input_bytes) / total_seconds : 0.0f;
        float output_bps = total_seconds > 0 ? static_cast<float>(model_output_bytes) / total_seconds : 0.0f;

        float throughput = 1.0f / avg_latency;

        if (verbose) {
            std::cout << "\n *~~~~~~~~~ Metrics ~~~~~~~~~*\n";
            std::cout << " | Overall Accuracy: " << accuracy * 100.0f << "%\n";
            std::cout << " | - - - - - - - - - - - -" << std::endl;
            std::cout << " | Macro Precision:     " << macro_precision * 100.0f << "%\n";
            std::cout << " | Macro Recall:        " << macro_recall * 100.0f << "%\n";
            std::cout << " | Macro F1 Score:      " << macro_f1 * 100.0f << "%\n";
            std::cout << " | - - - - - - - - - - - -" << std::endl;
            std::cout << " | Weighted Precision:  " << weighted_precision * 100.0f << "%\n";
            std::cout << " | Weighted Recall:     " << weighted_recall * 100.0f << "%\n";
            std::cout << " | Weighted F1 Score:   " << weighted_f1 * 100.0f << "%\n";
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

            std::cout << "\n *~~~~ Per-class Metrics ~~~~*\n";
            for (size_t c = 0; c < num_classes; ++c) {
                std::cout << " | Class " << c << ":\n";
                std::cout << " |   Precision: " << class_precision[c] * 100.0f << "%\n";
                std::cout << " |   Recall: " << class_recall[c] * 100.0f << "%\n";
                std::cout << " |   F1 Score: " << class_f1[c] * 100.0f << "%\n";
            }
        }
    }
}
