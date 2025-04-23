#pragma once
#include "../tensor.hpp"
#include <random>
#include <cmath>
#include <iostream>

namespace Thot {

    enum class Initialization {
        Zeros,
        Ones,
        Uniform,
        Normal,
        Xavier,
        He,          // Good for ReLU activation layers
        LeCun        // Scaled for the fan-in size
    };

    namespace Initializers {

        inline std::mt19937& get_random_generator() {
            static thread_local std::random_device rd;
            static thread_local std::mt19937 gen(rd());
            return gen;
        }

        inline void initialize_tensor(Utils::Tensor& tensor, Initialization method, int fan_in = 0, int fan_out = 0) {
            std::vector<int> shape = tensor.shape();
            size_t size = tensor.size();

            if (fan_in == 0 || fan_out == 0) {
                if (shape.size() >= 2) {
                    fan_out = shape[0];
                    fan_in = shape[1];

                    // For convolutional kernels: [out_channels, in_channels, height, width]
                    if (shape.size() > 2) {
                        for (size_t i = 2; i < shape.size(); ++i) {
                            fan_in *= shape[i];
                        }
                    }
                }
                else {
                    fan_in = fan_out = static_cast<int>(size);
                }
            }

            std::vector<float> host_data(size);

            switch (method) {
            case Initialization::Zeros:
                std::fill(host_data.begin(), host_data.end(), static_cast<float>(0));
                break;

            case Initialization::Ones:
                std::fill(host_data.begin(), host_data.end(), static_cast<float>(1));
                break;

            case Initialization::Uniform: {
                float range = static_cast<float>(0.1);
                std::uniform_real_distribution<float> dist(-range, range);
                auto& gen = get_random_generator();
                for (size_t i = 0; i < size; ++i) {
                    host_data[i] = static_cast<float>(dist(gen));
                }
                break;
            }

            case Initialization::Normal: {
                float stddev = static_cast<float>(0.01);
                std::normal_distribution<float> dist(0.0f, stddev);
                auto& gen = get_random_generator();
                for (size_t i = 0; i < size; ++i) {
                    host_data[i] = static_cast<float>(dist(gen));
                }
                break;
            }

            case Initialization::Xavier: {
                // Xavier/Glorot: scale = sqrt(6 / (fan_in + fan_out))
                float scale = std::sqrt(6.0f / (fan_in + fan_out));
                std::uniform_real_distribution<float> dist(-scale, scale);
                auto& gen = get_random_generator();
                for (size_t i = 0; i < size; ++i) {
                    host_data[i] = static_cast<float>(dist(gen));
                }
                break;
            }

            case Initialization::He: {
                // He initialization: scale = sqrt(2 / fan_in)
                float scale = std::sqrt(2.0f / fan_in);
                std::normal_distribution<float> dist(0.0f, scale);
                auto& gen = get_random_generator();
                for (size_t i = 0; i < size; ++i) {
                    host_data[i] = static_cast<float>(dist(gen));
                }
                break;
            }

            case Initialization::LeCun: {
                // LeCun initialization: scale = sqrt(1 / fan_in)
                float scale = std::sqrt(1.0f / fan_in);
                std::normal_distribution<float> dist(0.0f, scale);
                auto& gen = get_random_generator();
                for (size_t i = 0; i < size; ++i) {
                    host_data[i] = static_cast<float>(dist(gen));
                }
                break;
            }
            }

            tensor.upload(host_data);



        }

        inline void zeros(Utils::Tensor& tensor) {
            initialize_tensor(tensor, Initialization::Zeros);
        }

        inline void ones(Utils::Tensor& tensor) {
            initialize_tensor(tensor, Initialization::Ones);
        }

        inline void uniform(Utils::Tensor& tensor, float low = -0.1, float high = 0.1) {
            std::vector<int> shape = tensor.shape();
            size_t size = tensor.size();
            std::vector<float> host_data(size);

            std::uniform_real_distribution<float> dist(static_cast<float>(low), static_cast<float>(high));
            auto& gen = get_random_generator();
            for (size_t i = 0; i < size; ++i) {
                host_data[i] = static_cast<float>(dist(gen));
            }

            tensor.upload(host_data);
        }

        inline void normal(Utils::Tensor& tensor, float mean = 0.0, float stddev = 0.01) {
            std::vector<int> shape = tensor.shape();
            size_t size = tensor.size();
            std::vector<float> host_data(size);

            std::normal_distribution<float> dist(static_cast<float>(mean), static_cast<float>(stddev));
            auto& gen = get_random_generator();
            for (size_t i = 0; i < size; ++i) {
                host_data[i] = static_cast<float>(dist(gen));
            }

            tensor.upload(host_data);
        }

        inline void xavier(Utils::Tensor& tensor, int fan_in = 0, int fan_out = 0) {
            initialize_tensor(tensor, Initialization::Xavier, fan_in, fan_out);
        }

        inline void he(Utils::Tensor& tensor, int fan_in = 0) {
            initialize_tensor(tensor, Initialization::He, fan_in);
        }

        inline void lecun(Utils::Tensor& tensor, int fan_in = 0) {
            initialize_tensor(tensor, Initialization::LeCun, fan_in);
        }

        inline std::string to_string(Initialization init) {
            switch (init) {
            case Initialization::Zeros: return "Zeros";
            case Initialization::Ones: return "Ones";
            case Initialization::Uniform: return "Uniform";
            case Initialization::Normal: return "Normal";
            case Initialization::Xavier: return "Xavier";
            case Initialization::He: return "He";
            case Initialization::LeCun: return "LeCun";
            default: return "Unknown";
            }
        }
    } // namespace Initializers
} // namespace Thot