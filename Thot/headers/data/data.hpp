#pragma once
#include <random>
#include <cmath>
#include <vector>
#include <iostream>
#include <limits>
#include <algorithm>
#include <utility>
#include "details/mnist.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Thot {

    enum class DataType {
        Linear,
        Quadratic,
        Sine,
        Circle,
        Spiral,
        XOR,
        Blobs,
        MNIST
    };

    namespace Data {

        inline std::mt19937& get_random_generator() {
            static thread_local std::random_device rd;
            static thread_local std::mt19937 gen(rd());
            return gen;
        }

        inline std::string to_string(DataType type) {
            switch (type) {
            case DataType::Linear: return "Linear";
            case DataType::Quadratic: return "Quadratic";
            case DataType::Sine: return "Sine";
            case DataType::Circle: return "Circle";
            case DataType::Spiral: return "Spiral";
            case DataType::XOR: return "XOR";
            case DataType::Blobs: return "Blobs";
            case DataType::MNIST: return "MNIST";
            default: return "Unknown";
            }
        }

        inline void print_data_stats(const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& y) {
            if (X.empty() || y.empty()) {
                std::cout << "Empty dataset\n";
                return;
            }

            std::cout << "\nDataset Statistics:\n";
            std::cout << "------------------\n";
            std::cout << "Number of samples: " << X.size() << "\n";
            std::cout << "Number of features: " << X[0].size() << "\n";
            std::cout << "Number of targets: " << y[0].size() << "\n\n";

            std::cout << "Features Statistics:\n";
            for (size_t feature = 0; feature < X[0].size(); ++feature) {
                float min_val = std::numeric_limits<float>::max();
                float max_val = std::numeric_limits<float>::lowest();
                float sum = 0.0f;
                float sum_sq = 0.0f;

                for (const auto& sample : X) {
                    float val = sample[feature];
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                    sum += val;
                    sum_sq += val * val;
                }

                float mean = sum / X.size();
                float variance = (sum_sq / X.size()) - (mean * mean);
                float std_dev = std::sqrt(variance);

                std::cout << "Feature " << feature + 1 << ":\n";
                std::cout << "  Min: " << min_val << "\n";
                std::cout << "  Max: " << max_val << "\n";
                std::cout << "  Mean: " << mean << "\n";
                std::cout << "  Std Dev: " << std_dev << "\n";
                std::cout << "  Range: " << max_val - min_val << "\n\n";
            }

            std::cout << "Target Statistics:\n";
            for (size_t target = 0; target < y[0].size(); ++target) {
                float min_val = std::numeric_limits<float>::max();
                float max_val = std::numeric_limits<float>::lowest();
                float sum = 0.0f;
                float sum_sq = 0.0f;

                for (const auto& sample : y) {
                    float val = sample[target];
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                    sum += val;
                    sum_sq += val * val;
                }

                float mean = sum / y.size();
                float variance = (sum_sq / y.size()) - (mean * mean);
                float std_dev = std::sqrt(variance);

                std::cout << "Target " << target + 1 << ":\n";
                std::cout << "  Min: " << min_val << "\n";
                std::cout << "  Max: " << max_val << "\n";
                std::cout << "  Mean: " << mean << "\n";
                std::cout << "  Std Dev: " << std_dev << "\n";
                std::cout << "  Range: " << max_val - min_val << "\n\n";
            }
        }

        inline std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> generate_data(DataType type, int samples, float noise = 0.1, bool verbose = true) {
            std::vector<float> X;
            std::vector<float> y;
            auto& gen = get_random_generator();
            std::normal_distribution<float> noise_dist(0.0f, noise);

            switch (type) {
            case DataType::Linear: {
                X.resize(samples * 2);
                y.resize(samples);
                for (int i = 0; i < samples; ++i) {
                    float x = static_cast<float>(i) / samples;
                    X[i * 2] = x;
                    X[i * 2 + 1] = x;
                    y[i] = 2 * x + 1 + noise_dist(gen);
                }
                break;
            }

            case DataType::Quadratic: {
                X.resize(samples * 2);
                y.resize(samples);
                for (int i = 0; i < samples; ++i) {
                    float x = static_cast<float>(i) / samples;
                    X[i * 2] = x;
                    X[i * 2 + 1] = x * x;
                    y[i] = x * x + noise_dist(gen);
                }
                break;
            }

            case DataType::Sine: {
                X.resize(samples);
                y.resize(samples);
                for (int i = 0; i < samples; ++i) {
                    float x = static_cast<float>(i) / samples * 2 * M_PI;
                    X[i] = x;
                    y[i] = std::sin(x) + noise_dist(gen);
                }
                break;
            }

            case DataType::Circle: {
                X.resize(samples * 2);
                y.resize(samples);
                std::uniform_real_distribution<float> angle_dist(0, 2 * M_PI);
                for (int i = 0; i < samples; ++i) {
                    float angle = angle_dist(gen);
                    float radius = 1.0f + noise_dist(gen);
                    X[i * 2] = radius * std::cos(angle);
                    X[i * 2 + 1] = radius * std::sin(angle);
                    y[i] = (radius > 1.0f) ? 1.0f : 0.0f;
                }
                break;
            }

            case DataType::Spiral: {
                X.resize(samples * 2);
                y.resize(samples);
                for (int i = 0; i < samples; ++i) {
                    float t = static_cast<float>(i) / samples * 2 * M_PI;
                    float r = t / (2 * M_PI);
                    X[i * 2] = r * std::cos(t) + noise_dist(gen);
                    X[i * 2 + 1] = r * std::sin(t) + noise_dist(gen);
                    y[i] = (i < samples / 2) ? 1.0f : 0.0f;
                }
                break;
            }

            case DataType::XOR: {
                X.resize(samples * 2);
                y.resize(samples);
                std::uniform_real_distribution<float> dist(0, 1);
                for (int i = 0; i < samples; ++i) {
                    float x1 = dist(gen);
                    float x2 = dist(gen);
                    X[i * 2] = x1;
                    X[i * 2 + 1] = x2;
                    y[i] = (x1 < 0.5f && x2 < 0.5f) || (x1 >= 0.5f && x2 >= 0.5f) ? 0.0f : 1.0f;
                }
                break;
            }

            case DataType::Blobs: {
                X.resize(samples * 2);
                y.resize(samples);
                std::normal_distribution<float> blob1_x(0.0f, 0.5f);
                std::normal_distribution<float> blob1_y(0.0f, 0.5f);
                std::normal_distribution<float> blob2_x(1.0f, 0.5f);
                std::normal_distribution<float> blob2_y(1.0f, 0.5f);

                for (int i = 0; i < samples; ++i) {
                    if (i < samples / 2) {
                        X[i * 2] = blob1_x(gen);
                        X[i * 2 + 1] = blob1_y(gen);
                        y[i] = 0.0f;
                    }
                    else {
                        X[i * 2] = blob2_x(gen);
                        X[i * 2 + 1] = blob2_y(gen);
                        y[i] = 1.0f;
                    }
                }
                break;
            }

            case DataType::MNIST: {
                throw std::runtime_error("MNIST data must be loaded using load_mnist_train() or load_mnist_test()");
            }
            }

            std::vector<std::vector<float>> X_reshaped;
            std::vector<std::vector<float>> y_reshaped;

            int feature_count = samples > 0 ? static_cast<int>(X.size() / samples) : 0;
            for (int i = 0; i < samples; ++i) {
                std::vector<float> sample(feature_count);
                for (int f = 0; f < feature_count; ++f) {
                    sample[f] = X[i * feature_count + f];
                }
                X_reshaped.push_back(std::move(sample));
                y_reshaped.push_back({ y[i] });
            }

            if (verbose) {
                std::cout << "\nGenerated " << to_string(type) << " dataset with " << samples << " samples\n";
                print_data_stats(X_reshaped, y_reshaped);
            }

            return { X_reshaped, y_reshaped };
        }
    }
}