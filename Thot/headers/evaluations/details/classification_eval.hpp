#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <unordered_map>

namespace Evaluations {
    inline void evaluate_classification(const std::vector<std::vector<float>>& predictions, const std::vector<std::vector<float>>& targets, const std::vector<float>& latencies, size_t flops, bool verbose) {
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
        float avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / latencies.size();
        float throughput = 1.0f / avg_latency;

        if (verbose) {
            std::cout << "\nMetrics:\n";
            std::cout << "Overall Accuracy: " << accuracy * 100.0f << "%\n";
            std::cout << "Global Precision: " << global_precision * 100.0f << "%\n";
            std::cout << "Global Recall:    " << global_recall * 100.0f << "%\n";
            std::cout << "Global F1 Score:  " << global_f1 * 100.0f << "%\n";

            std::cout << "\nPer-class Metrics:\n";
            for (size_t c = 0; c < num_classes; ++c) {
                std::cout << "Class " << c << ":\n";
                std::cout << "  Precision: " << class_precision[c] * 100.0f << "%\n";
                std::cout << "  Recall: " << class_recall[c] * 100.0f << "%\n";
                std::cout << "  F1 Score: " << class_f1[c] * 100.0f << "%\n";
            }

            std::cout << "Average Latency: " << avg_latency << " ms\n";
            std::cout << "Throughput: " << throughput << " FLOPS\n";
        }
    }
}
