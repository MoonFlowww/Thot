#pragma once
#include "../tensor.hpp"

#include <random>
#include <cmath>
#include <iostream>

#include "../../cuda/cuh/initializations/random.cuh"

namespace Thot {
    enum class Initialization {
        Zeros,
        Ones,
        Uniform,
        Normal,
        Xavier,
        He,          // Good for ReLU activation layers
        LeCun,        // Scaled for the fan-in size
        //Orthogonal,
        TruncatedNormal,
        Dirac,
        Lyapunov,
        None
    };

    inline Initialization initialization_from_string(const std::string &name) {
        if (name == "Zeros") return Initialization::Zeros;
        if (name == "Ones") return Initialization::Ones;
        if (name == "Uniform") return Initialization::Uniform;
        if (name == "Normal") return Initialization::Normal;
        if (name == "Xavier") return Initialization::Xavier;
        if (name == "He") return Initialization::He;
        if (name == "LeCun") return Initialization::LeCun;
        if (name == "TruncatedNormal") return Initialization::TruncatedNormal;
        if (name == "Dirac") return Initialization::Dirac;
        //if (name == "Orthogonal") return Initialization::Orthogonal;
        if (name == "Lyapunov") return Initialization::Lyapunov;
        if (name == "None") return Initialization::None;
        throw std::runtime_error("Unknown initialization: " + name);
    }

    namespace Initializations {
        inline std::string to_string(Initialization init) {
            switch (init) {
                case Initialization::Zeros: return "Zeros";
                case Initialization::Ones: return "Ones";
                case Initialization::Uniform: return "Uniform";
                case Initialization::Normal: return "Normal";
                case Initialization::Xavier: return "Xavier";
                case Initialization::He: return "He";
                case Initialization::LeCun: return "LeCun";
                case Initialization::TruncatedNormal: return "TruncatedNormal";
                case Initialization::Dirac: return "Dirac";
                    //case Initialization::Orthogonal: return "Orthogonal";
                case Initialization::Lyapunov: return "Lyapunov";
                case Initialization::None: return "None";
                default: return "Unknown";
            }
        }

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

#ifdef __CUDACC__
            switch (method) {
                case Initialization::Zeros:
                    cudaMemset(tensor.data(), 0, size * sizeof(float));
                    break;
                case Initialization::Ones:
                    cuda::initializations::launchFill(tensor.data(), static_cast<int>(size), 1.0f);
                    break;
                case Initialization::Uniform: {
                    float range = 0.1f;
                    cuda::initializations::launchRandomUniform(tensor.data(), static_cast<int>(size), -range, range);
                    break;
                }
                case Initialization::Normal: {
                    float stddev = 0.01f;
                    cuda::initializations::launchRandomNormal(tensor.data(), static_cast<int>(size), 0.0f, stddev);
                    break;
                }
                case Initialization::Xavier: {
                    float scale = std::sqrt(6.0f / (fan_in + fan_out));
                    cuda::initializations::launchRandomUniform(tensor.data(), static_cast<int>(size), -scale, scale);
                    break;
                }
                case Initialization::He: {
                    float scale = std::sqrt(2.0f / fan_in);
                    cuda::initializations::launchRandomNormal(tensor.data(), static_cast<int>(size), 0.0f, scale);
                    break;
                }
                case Initialization::LeCun: {
                    float scale = std::sqrt(1.0f / fan_in);
                    cuda::initializations::launchRandomNormal(tensor.data(), static_cast<int>(size), 0.0f, scale);
                    break;
                }
                case Initialization::TruncatedNormal: {
                    float stddev = 0.01f;
                    cuda::initializations::launchRandomTruncatedNormal(tensor.data(), static_cast<int>(size), 0.0f, stddev);
                    break;
                }
                case Initialization::Dirac: {
                    if (shape.size() == 2) {
                        cuda::initializations::launchDirac(tensor.data(), shape[0], shape[1]);
                    } else {
                        cuda::initializations::launchFill(tensor.data(), static_cast<int>(size), 0.0f);
                    }
                    break;
                }
                case Initialization::Lyapunov: {
                    if (shape.size() == 2) {
                        cuda::initializations::launchLyapunov(tensor.data(), shape[0], shape[1]);
                    } else {
                        cuda::initializations::launchRandomNormal(tensor.data(), static_cast<int>(size), 0.0f, 1.0f);
                    }
                    break;
                }
                default:
                    cuda::initializations::launchFill(tensor.data(), static_cast<int>(size), 0.0f);
                    break;
            }
            cudaDeviceSynchronize();
            auto end_time = high_resolution_clock::now();
            double ms = duration<double, std::milli>(end_time - start_time).count();
            static double total_ms = 0.0;
            static size_t total_params = 0;
            total_ms += ms;
            total_params += size;
            std::cout << "Initialized tensor with " << to_string(method)
                      << " in " << ms << " ms (" << (ms / size) << " ms/param). "
                      << "Total: " << total_ms << " ms over " << total_params << " params." << std::endl;
            return;
#endif
            std::vector<float> host_data(size);
            switch (method) {
                case Initialization::Zeros:
                    std::fill(host_data.begin(), host_data.end(), 0.0f);
                    break;
                case Initialization::Ones:
                    std::fill(host_data.begin(), host_data.end(), 1.0f);
                    break;
                case Initialization::Uniform: {
                    float range = 0.1f;
                    std::uniform_real_distribution<float> dist(-range, range);
                    auto& gen = get_random_generator();
                    for (size_t i = 0; i < size; ++i) host_data[i] = dist(gen);
                    break;
                }
                case Initialization::Normal: {
                    float stddev = 0.01f;
                    std::normal_distribution<float> dist(0.0f, stddev);
                    auto& gen = get_random_generator();
                    for (size_t i = 0; i < size; ++i) host_data[i] = dist(gen);
                    break;
                }
                case Initialization::Xavier: {
                    float scale = std::sqrt(6.0f / (fan_in + fan_out));
                    std::uniform_real_distribution<float> dist(-scale, scale);
                    auto& gen = get_random_generator();
                    for (size_t i = 0; i < size; ++i) host_data[i] = dist(gen);
                    break;
                }
                case Initialization::He: {
                    float scale = std::sqrt(2.0f / fan_in);
                    std::normal_distribution<float> dist(0.0f, scale);
                    auto& gen = get_random_generator();
                    for (size_t i = 0; i < size; ++i) host_data[i] = dist(gen);
                    break;
                }
                case Initialization::LeCun: {
                    float scale = std::sqrt(1.0f / fan_in);
                    std::normal_distribution<float> dist(0.0f, scale);
                    auto& gen = get_random_generator();
                    for (size_t i = 0; i < size; ++i) host_data[i] = dist(gen);
                    break;
                }
                case Initialization::TruncatedNormal: {
                    float stddev = 0.01f;
                    auto& gen = get_random_generator();
                    std::normal_distribution<float> dist(0.0f, stddev);
                    for (size_t i = 0; i < size; ++i) {
                        float val;
                        do { val = dist(gen); } while (std::abs(val) > 2 * stddev);
                        host_data[i] = val;
                    }
                    break;
                }
                case Initialization::Dirac: {
                    std::fill(host_data.begin(), host_data.end(), 0.0f);
                    if (shape.size() == 2) {
                        int diag = std::min(shape[0], shape[1]);
                        for (int i = 0; i < diag; ++i) host_data[i * shape[1] + i] = 1.0f;
                    }
                    break;
                }
                case Initialization::Lyapunov: {
                    std::normal_distribution<float> dist(0.0f, 1.0f);
                    auto& gen = get_random_generator();
                    for (size_t i = 0; i < size; ++i) host_data[i] = dist(gen);

                    if (shape.size() == 2) {
                        int rows = shape[0];
                        int cols = shape[1];
                        std::vector<float> v(cols, 1.0f);
                        for (int iter = 0; iter < 20; ++iter) {
                            std::vector<float> v_new(rows, 0.0f);
                            for (int r = 0; r < rows; ++r) {
                                for (int c = 0; c < cols; ++c) v_new[r] += host_data[r * cols + c] * v[c];
                            }
                            float norm = 0.0f;
                            for (float val : v_new) norm += val * val;
                            norm = std::sqrt(norm);
                            if (norm > 0) {
                                for (float& val : v_new) val /= norm;
                            }
                            v = v_new;
                        }
                        float spectral_radius = 0.0f;
                        for (int r = 0; r < rows; ++r) {
                            float acc = 0.0f;
                            for (int c = 0; c < cols; ++c) acc += host_data[r * cols + c] * v[c];
                            spectral_radius += acc * v[r];
                        }
                        spectral_radius = std::abs(spectral_radius);
                        if (spectral_radius > 0) {
                            float scale = 0.95f / spectral_radius;
                            for (size_t i = 0; i < size; ++i) host_data[i] *= scale;
                        }
                    }
                    break;
                }
                default:
                    std::fill(host_data.begin(), host_data.end(), 0.0f);
                    break;
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
            size_t size = tensor.size();
#ifdef __CUDACC__
            cuda::initializations::launchRandomUniform(tensor.data(), static_cast<int>(size), low, high);
#else
            std::vector<float> host_data(size);
            std::uniform_real_distribution<float> dist(static_cast<float>(low), static_cast<float>(high));
            auto& gen = get_random_generator();
            for (size_t i = 0; i < size; ++i) {
                host_data[i] = static_cast<float>(dist(gen));
            }
            tensor.upload(host_data);
#endif
        }

        inline void normal(Utils::Tensor& tensor, float mean = 0.0, float stddev = 0.01) {
            size_t size = tensor.size();
#ifdef __CUDACC__
            cuda::initializations::launchRandomNormal(tensor.data(), static_cast<int>(size), mean, stddev);
#else
            std::vector<float> host_data(size);
            std::normal_distribution<float> dist(static_cast<float>(mean), static_cast<float>(stddev));
            auto& gen = get_random_generator();
            for (size_t i = 0; i < size; ++i) {
                host_data[i] = static_cast<float>(dist(gen));
            }
            tensor.upload(host_data);
#endif
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

        inline void TruncatedNormal(Utils::Tensor& tensor, int fan_in = 0) {
            initialize_tensor(tensor, Initialization::TruncatedNormal, fan_in);
        }
        /*
        inline void Orthogonal(Utils::Tensor& tensor, int fan_in = 0) {
            initialize_tensor(tensor, Initialization::Orthogonal, fan_in);
        }
        */
        inline void Dirac(Utils::Tensor& tensor, int fan_in = 0) {
            initialize_tensor(tensor, Initialization::Dirac, fan_in);
        }

        inline void Lyapunov(Utils::Tensor& tensor, int fan_in = 0) {
            initialize_tensor(tensor, Initialization::Lyapunov, fan_in);
        }
    }
}