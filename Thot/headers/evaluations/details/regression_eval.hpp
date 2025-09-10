#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include "utils/translators.hpp"

namespace Evaluations {
    inline void evaluate_regression(const std::vector<std::vector<float>>& predictions, const std::vector<std::vector<float>>& targets, const  std::vector<float>& latencies,
        size_t flops, size_t input_size, size_t output_size, size_t total_flops, size_t total_parm, bool verbose) {
        if (verbose) {
            std::cout << "\nRegression Evaluation:\n";
            std::cout << "---------------------\n";
        }

        float mse = 0.0f;
        float mae = 0.0f;
        float r2 = 0.0f;

        for (size_t i = 0; i < predictions.size(); ++i) {
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

        float total_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0f);
        float avg_latency = total_latency / latencies.size();
        float max_latency = *std::max_element(latencies.begin(), latencies.end());
        float min_latency = *std::min_element(latencies.begin(), latencies.end());
        float sq_sum = std::inner_product(latencies.begin(), latencies.end(), latencies.begin(), 0.0f);
        float std_latency = std::sqrt(sq_sum / latencies.size() - avg_latency * avg_latency);
        float skew_latency = 0.0f;
        for (float l : latencies) skew_latency += std::pow(l - avg_latency, 3);
        skew_latency /= latencies.size();
        skew_latency /= std::pow(std_latency, 3) + 1e-10f;
        std::unordered_map<int,int> freq;
        for (float l : latencies) freq[static_cast<int>(std::round(l))]++;
        float mode_latency = avg_latency - (std_latency*skew_latency)/2; // pearson approx

        size_t model_input_bytes = predictions.size() * input_size * sizeof(float);
        size_t model_output_bytes = predictions.size() * output_size * sizeof(float);
        float total_seconds = total_latency / 1000.0f;
        float input_bps  = total_seconds > 0 ? static_cast<float>(model_input_bytes) / total_seconds : 0.0f;
        float output_bps = total_seconds > 0 ? static_cast<float>(model_output_bytes) / total_seconds : 0.0f;
        float forward_s = 1.0f / avg_latency;

        if (verbose) {
            std::cout << "\n *~~~~~~~~~ Metrics ~~~~~~~~~*\n";

            std::cout << " | MSE: " << mse << "\n";
            std::cout << " | MAE: " << mae << "\n";
            std::cout << " | R-squared: " << r2 << "\n";
            std::cout << " *~~~~~~~~~~~~~~~~~~~~~~~~~~~~*" << std::endl;
            std::cout << " | Latency Average : " << Thot::format_time(avg_latency) << "\n";
            std::cout << " | Latency Std Dev: " << Thot::format_time(std_latency) << "\n";
            std::cout << " | Latency Skew: " << skew_latency << "\n";
            std::cout << " | Latency Mode: " << Thot::format_time(mode_latency) << "\n";
            std::cout << " | Latency +: " << Thot::format_time(max_latency) << "\n";
            std::cout << " | Latency -: " << Thot::format_time(min_latency) << "\n";
            std::cout << " *~~~~~~~~~~~~~~~~~~~~~~~~~~~~*" << std::endl;

            std::cout << " | Input Bytes/s: " << Thot::formatBytes(input_bps) << "\n";
            std::cout << " | Output Bytes/s: " << Thot::formatBytes(output_bps) << "\n";
            std::cout << " | Throughput/s: " << Thot::formatBytes(input_bps+output_bps) << "\n";
            std::cout << " | Forward/s: " << Thot::human_readable_size(static_cast<size_t>(forward_s)) << "\n";

            std::cout << " *- - - - - - -  - - - - - - -*" << std::endl;
            std::cout << " | Arithmetic Intensity: " << static_cast<float>(total_flops)/(input_bps+output_bps) << "  SweetSpot∈[5;20]\n";
            std::cout << " | FLOP/s: " << Thot::human_readable_size(total_flops*static_cast<size_t>(forward_s)) << "\n";
            std::cout << " | Throughput/Parm: " << Thot::human_readable_size((input_bps+output_bps)/static_cast<float>(total_parm)) << "\n";
            std::cout << " *~~~~~~~~~~~~~~~~~~~~~~~~~~~~*" << std::endl;

        }
    }
}