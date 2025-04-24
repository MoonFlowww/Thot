#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace Evaluations {
    inline void evaluate_timeseries(const std::vector<std::vector<float>>& predictions, const std::vector<std::vector<float>>& targets, const  std::vector<float>& latencies, size_t flops, bool verbose ) {
        if (verbose) {
            std::cout << "\nTime Series Evaluation:\n";
            std::cout << "----------------------\n";
        }

        float mse = 0.0f;
        float mae = 0.0f;
        float r2 = 0.0f;

        for (size_t i = 0; i < predictions.size(); ++i) {
            if (verbose) {
                std::cout << "Time Step " << i << ":\n";
                std::cout << "   Predicted: [";
                for (float x : predictions[i]) std::cout << x << " ";
                std::cout << "]\n   Actual: [";
                for (float y : targets[i]) std::cout << y << " ";
                std::cout << "]\n" << std::endl;
            }

            float sum_squared_error = 0.0f;
            float sum_absolute_error = 0.0f;
            float sum_squared_total = 0.0f;
            float mean_target = 0.0f;

            for (size_t j = 0; j < predictions[i].size(); ++j) {
                float error = predictions[i][j] - targets[i][j];
                sum_squared_error += error * error;
                sum_absolute_error += std::abs(error);
                mean_target += targets[i][j];
            }

            mean_target /= predictions[i].size();

            for (size_t j = 0; j < targets[i].size(); ++j) {
                float diff = targets[i][j] - mean_target;
                sum_squared_total += diff * diff;
            }

            mse += sum_squared_error / predictions[i].size();
            mae += sum_absolute_error / predictions[i].size();
            r2 += 1.0f - (sum_squared_error / (sum_squared_total + 1e-10f));
        }

        mse /= predictions.size();
        mae /= predictions.size();
        r2 /= predictions.size();

        float avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / latencies.size();
        float throughput = 1.0f / avg_latency;

        if (verbose) {
            std::cout << "\nMetrics:\n";
            std::cout << "Mean Squared Error: " << mse << "\n";
            std::cout << "Mean Absolute Error: " << mae << "\n";
            std::cout << "R-squared: " << r2 << "\n";
            std::cout << "Average Latency: " << avg_latency << " ms\n";
            std::cout << "Throughput: " << throughput << " FLOPS\n";
        }
    }
}