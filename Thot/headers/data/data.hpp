#pragma once
#include <random>
#include <cmath>
#include <vector>
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

        inline std::pair<std::vector<float>, std::vector<float>> generate_data(DataType type, int samples, float noise = 0.1) {
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

            return { X, y };
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
    }
}