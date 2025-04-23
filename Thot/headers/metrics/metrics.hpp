#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <iostream>
#include "../tensor.hpp"

namespace Thot {
    class Metrics {
    public:
        struct PerformanceMetrics {
            float accuracy;
            float precision;
            float recall;
            float f1_score;
            float throughput;
            size_t flops;
        };

        struct ParetoPoint {
            float accuracy;
            float throughput;
            size_t flops;
        };

        static PerformanceMetrics compute_metrics(
            const std::vector<std::vector<float>>& predictions,
            const std::vector<std::vector<float>>& targets,
            const std::vector<float>& latencies,
            size_t flops
        ) {
            PerformanceMetrics metrics;

            float total_accuracy = 0.0f;
            for (size_t i = 0; i < predictions.size(); ++i) {
                total_accuracy += compute_accuracy(predictions[i], targets[i]);
            }
            metrics.accuracy = total_accuracy / predictions.size();

            metrics.precision = compute_precision(predictions, targets);
            metrics.recall = compute_recall(predictions, targets);
            metrics.f1_score = compute_f1_score(metrics.precision, metrics.recall);

            metrics.throughput = compute_throughput(latencies);
            metrics.flops = flops;

            return metrics;
        }

        static std::vector<ParetoPoint> compute_pareto_frontier(
            const std::vector<std::vector<float>>& predictions,
            const std::vector<std::vector<float>>& targets,
            const std::vector<float>& latencies,
            size_t flops
        ) {
            std::vector<ParetoPoint> points;
            points.reserve(predictions.size());

            for (size_t i = 0; i < predictions.size(); ++i) {
                float accuracy = compute_accuracy(predictions[i], targets[i]);
                float throughput = compute_throughput({ latencies[i] });
                points.push_back({ accuracy, throughput, flops });
            }

            std::sort(points.begin(), points.end(),
                [](const ParetoPoint& a, const ParetoPoint& b) {
                    return a.accuracy > b.accuracy;
                });

            std::vector<ParetoPoint> frontier;
            float max_throughput = 0.0f;

            for (const auto& point : points) {
                if (point.throughput > max_throughput) {
                    frontier.push_back(point);
                    max_throughput = point.throughput;
                }
            }

            return frontier;
        }

        static void print_metrics(const PerformanceMetrics& metrics, const std::vector<ParetoPoint>& frontier) {
            std::cout << "\nPerformance Metrics:\n";
            std::cout << "-------------------\n";
            std::cout << "Accuracy: " << metrics.accuracy * 100.0f << "%\n";
            std::cout << "Precision: " << metrics.precision * 100.0f << "%\n";
            std::cout << "Recall: " << metrics.recall * 100.0f << "%\n";
            std::cout << "F1 Score: " << metrics.f1_score * 100.0f << "%\n";
            std::cout << "Throughput: " << metrics.throughput << " samples/second\n";
            std::cout << "Model FLOPs: " << metrics.flops << "\n";

            if (!frontier.empty()) {
                std::cout << "\nPareto Frontier Points:\n";
                std::cout << "Accuracy\tThroughput\tFLOPs\n";
                for (const auto& point : frontier) {
                    std::cout << point.accuracy * 100.0f << "%\t\t"
                        << point.throughput << "\t\t"
                        << point.flops << "\n";
                }
            }
        }

    private:
        static float compute_accuracy(const std::vector<float>& predictions, const std::vector<float>& targets) {
            if (predictions.size() != targets.size()) return 0.0f;

            size_t correct = 0;
            for (size_t i = 0; i < predictions.size(); ++i) {
                float pred = predictions[i] > 0.5f ? 1.0f : 0.0f;
                if (pred == targets[i]) correct++;
            }

            return static_cast<float>(correct) / predictions.size();
        }

        static float compute_throughput(const std::vector<float>& latencies) {
            if (latencies.empty()) return 0.0f;
            float avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / latencies.size();
            return 1.0f / avg_latency;
        }

        static float compute_precision(const std::vector<std::vector<float>>& predictions,
            const std::vector<std::vector<float>>& targets) {
            float true_positives = 0.0f;
            float false_positives = 0.0f;

            for (size_t i = 0; i < predictions.size(); ++i) {
                for (size_t j = 0; j < predictions[i].size(); ++j) {
                    float pred = predictions[i][j] >= 0.5f ? 1.0f : 0.0f;
                    float target = targets[i][j];

                    if (pred == 1.0f && target == 1.0f) true_positives += 1.0f;
                    else if (pred == 1.0f && target == 0.0f) false_positives += 1.0f;
                }
            }

            return true_positives / (true_positives + false_positives + 1e-10f);
        }

        static float compute_recall(const std::vector<std::vector<float>>& predictions,
            const std::vector<std::vector<float>>& targets) {
            float true_positives = 0.0f;
            float false_negatives = 0.0f;

            for (size_t i = 0; i < predictions.size(); ++i) {
                for (size_t j = 0; j < predictions[i].size(); ++j) {
                    float pred = predictions[i][j] >= 0.5f ? 1.0f : 0.0f;
                    float target = targets[i][j];

                    if (pred == 1.0f && target == 1.0f) true_positives += 1.0f;
                    else if (pred == 0.0f && target == 1.0f) false_negatives += 1.0f;
                }
            }

            return true_positives / (true_positives + false_negatives + 1e-10f);
        }

        static float compute_f1_score(float precision, float recall) {
            return 2.0f * (precision * recall) / (precision + recall + 1e-10f);
        }
    };
}