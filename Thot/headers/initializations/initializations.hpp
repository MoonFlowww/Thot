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
                    for (int i = 0; i < diag; ++i) {
                        host_data[i * shape[1] + i] = 1.0f;
                    }
                }
                break;
            }

            /*
            case Initialization::Orthogonal: {
                // Simplified: fill with normal then QR-decompose (requires linear algebra lib)
                // Placeholder: same as normal
                float stddev = 1.0f;
                std::normal_distribution<float> dist(0.0f, stddev);
                auto& gen = get_random_generator();
                for (size_t i = 0; i < size; ++i) host_data[i] = dist(gen);
                // TODO: apply QR for true orthogonality
                break;
                }
                */

            case Initialization::Lyapunov: {
                // Generate random Gaussian weights
                std::normal_distribution<float> dist(0.0f, 1.0f);
                auto& gen = get_random_generator();
                for (size_t i = 0; i < size; ++i) {
                    host_data[i] = dist(gen);
                }

                // If 2D (matrix), enforce spectral radius < 1
                if (shape.size() == 2) {
                    int rows = shape[0];
                    int cols = shape[1];

                    // crude power iteration to approximate spectral radius
                    std::vector<float> v(cols, 1.0f);
                    for (int iter = 0; iter < 20; ++iter) {
                        std::vector<float> v_new(rows, 0.0f);
                        for (int r = 0; r < rows; ++r) {
                            for (int c = 0; c < cols; ++c) {
                                v_new[r] += host_data[r * cols + c] * v[c];
                            }
                        }
                        float norm = 0.0f;
                        for (float val : v_new) norm += val * val;
                        norm = std::sqrt(norm);
                        if (norm > 0) {
                            for (float& val : v_new) val /= norm;
                        }
                        v = v_new;
                    }

                    // Rayleigh quotient estimate of spectral radius
                    float spectral_radius = 0.0f;
                    for (int r = 0; r < rows; ++r) {
                        float acc = 0.0f;
                        for (int c = 0; c < cols; ++c) {
                            acc += host_data[r * cols + c] * v[c];
                        }
                        spectral_radius += acc * v[r];
                    }
                    spectral_radius = std::abs(spectral_radius);

                    if (spectral_radius > 0) {
                        float scale = 0.95f / spectral_radius; // ensure < 1
                        for (size_t i = 0; i < size; ++i) {
                            host_data[i] *= scale;
                        }
                    }
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



    } // namespace Initializers
} // namespace Thot