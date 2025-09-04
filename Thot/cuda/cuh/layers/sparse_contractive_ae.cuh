#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cmath>

namespace cuda {
    namespace layers {

        inline float computeKLSparsity(const float* activations, int batch_size, int hidden_size, float rho) {
            std::vector<float> host(batch_size * hidden_size);
            cudaMemcpy(host.data(), activations, host.size() * sizeof(float), cudaMemcpyDeviceToHost);
            float kl = 0.0f;
            for (int j = 0; j < hidden_size; ++j) {
                float rho_hat = 0.0f;
                for (int b = 0; b < batch_size; ++b) {
                    rho_hat += host[b * hidden_size + j];
                }
                rho_hat /= static_cast<float>(batch_size);
                rho_hat = std::min(std::max(rho_hat, 1e-6f), 1.0f - 1e-6f);
                kl += rho * std::log(rho / rho_hat) + (1.0f - rho) * std::log((1.0f - rho) / (1.0f - rho_hat));
            }
            return kl;
        }

        inline float computeContractiveLoss(const float* activations, const float* weights,
                                            int batch_size, int input_size, int hidden_size) {
            std::vector<float> act(batch_size * hidden_size);
            std::vector<float> w(hidden_size * input_size);
            cudaMemcpy(act.data(), activations, act.size() * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(w.data(), weights, w.size() * sizeof(float), cudaMemcpyDeviceToHost);
            float loss = 0.0f;
            for (int j = 0; j < hidden_size; ++j) {
                float mean = 0.0f;
                for (int b = 0; b < batch_size; ++b) {
                    mean += act[b * hidden_size + j];
                }
                mean /= static_cast<float>(batch_size);
                float term = mean * (1.0f - mean);
                term = term * term;
                for (int i = 0; i < input_size; ++i) {
                    float wji = w[j * input_size + i];
                    loss += term * wji * wji;
                }
            }
            return loss;
        }

    } // namespace layers
} // namespace cuda
