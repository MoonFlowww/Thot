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

    inline float dtw_distance(const std::vector<float>& seq1, const std::vector<float>& seq2) {
        size_t n = seq1.size();
        size_t m = seq2.size();
        std::vector<std::vector<float>> dp(n + 1, std::vector<float>(m + 1, 1e9f));
        dp[0][0] = 0.0f;
        for (size_t i = 1; i <= n; ++i) {
            for (size_t j = 1; j <= m; ++j) {
                float cost = std::abs(seq1[i - 1] - seq2[j - 1]);
                dp[i][j] = cost + std::min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]});
            }
        }
        return dp[n][m];
    }

    inline float corr(const std::vector<float>& x, const std::vector<float>& y) {
        size_t n = x.size();
        float mean_x = std::accumulate(x.begin(), x.end(), 0.0f) / n;
        float mean_y = std::accumulate(y.begin(), y.end(), 0.0f) / n;

        float num = 0.0f, den_x = 0.0f, den_y = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            float dx = x[i] - mean_x;
            float dy = y[i] - mean_y;
            num += dx * dy;
            den_x += dx * dx;
            den_y += dy * dy;
        }
        return num / (std::sqrt(den_x * den_y) + 1e-10f);
    }

    inline void evaluate_timeseries(const std::vector<std::vector<float>>& predictions, const std::vector<std::vector<float>>& targets, const  std::vector<float>& latencies,
        size_t flops, size_t input_size, size_t output_size, size_t total_flops, size_t total_parm, bool verbose = false ) {
        if (verbose) {
            std::cout << "\nTime Series Evaluation:\n";
            std::cout << "----------------------\n";
        }

        float mse = 0.0f;
        float mae = 0.0f;
        float r2 = 0.0f;
        float total_dtw = 0.0f;

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

            total_dtw += dtw_distance(predictions[i], targets[i]);
        }

        mse /= predictions.size();
        mae /= predictions.size();
        r2 /= predictions.size();
        float avg_dtw = total_dtw / predictions.size();

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

        //TODO: Add DTW, msIC, msIR
        // DTW: Dyc Time Wrapping
        // msIC: mean sequencial correlation = (1/(Batchs*Output)) * Batchs_Sum( Outputs_Sum( Cov/Sigma))
        // msIR: correlation stability ratio = msIC/TrackingError(msIC_i;msIC)
        if (verbose) {
            std::cout << "\n *~~~~~~~~~ Metrics ~~~~~~~~~*\n";

            std::cout << " | Mean Squared Error: " << mse << "\n";
            std::cout << " | Mean Absolute Error: " << mae << "\n";
            std::cout << " | R-squared: " << r2 << "\n";
            std::cout << " | DTW: " << avg_dtw << "\n";
            std::cout << " | msIR: " << "coming soon..." << "\n";
            std::cout << " | msIC: " << "coming soon..." << "\n";
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