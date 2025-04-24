#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace Evaluations {
    inline void evaluate_classification(const std::vector<std::vector<float>>& predictions, const std::vector<std::vector<float>>& targets, const  std::vector<float>& latencies, size_t flops, bool verbose) {
        if (verbose) {
            std::cout << "\nMulti-class Classification Evaluation:\n";
            std::cout << "------------------------------------\n";
        }

        float accuracy = 0.0f;
        std::vector<float> class_precision;
        std::vector<float> class_recall;
        std::vector<float> class_f1;

        size_t num_classes = predictions[0].size();
        class_precision.resize(num_classes, 0.0f);
        class_recall.resize(num_classes, 0.0f);
        class_f1.resize(num_classes, 0.0f);

        for (size_t i = 0; i < predictions.size(); ++i) {
            if (verbose) {
                std::cout << "Sample " << i << ":\n";
                std::cout << "   Predicted: [";
                for (float x : predictions[i]) std::cout << x << " ";
                std::cout << "]\n   Actual: [";
                for (float y : targets[i]) std::cout << y << " ";
                std::cout << "]\n" << std::endl;
            }

            size_t pred_class = std::distance(predictions[i].begin(),
                std::max_element(predictions[i].begin(), predictions[i].end()));
            size_t actual_class = std::distance(targets[i].begin(),
                std::max_element(targets[i].begin(), targets[i].end()));

            if (pred_class == actual_class) {
                accuracy += 1.0f;
            }

            for (size_t c = 0; c < num_classes; ++c) {
                float true_positives = 0.0f;
                float false_positives = 0.0f;
                float false_negatives = 0.0f;

                if (pred_class == c && actual_class == c) true_positives += 1.0f;
                else if (pred_class == c && actual_class != c) false_positives += 1.0f;
                else if (pred_class != c && actual_class == c) false_negatives += 1.0f;

                float precision = true_positives / (true_positives + false_positives + 1e-10f);
                float recall = true_positives / (true_positives + false_negatives + 1e-10f);
                float f1 = 2.0f * (precision * recall) / (precision + recall + 1e-10f);

                class_precision[c] += precision;
                class_recall[c] += recall;
                class_f1[c] += f1;
            }
        }

        accuracy /= predictions.size();
        for (size_t c = 0; c < num_classes; ++c) {
            class_precision[c] /= predictions.size();
            class_recall[c] /= predictions.size();
            class_f1[c] /= predictions.size();
        }

        float avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / latencies.size();
        float throughput = 1.0f / avg_latency;

        if (verbose) {
            std::cout << "\nMetrics:\n";
            std::cout << "Overall Accuracy: " << accuracy * 100.0f << "%\n";
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