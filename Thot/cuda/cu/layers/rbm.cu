#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cmath>
#include <curand_kernel.h>
#include "../../cuh/layers/rbm.cuh"

namespace cuda {
    namespace layers {

        __global__ void rbm_visible_to_hidden(const float* visible, const float* weights, const float* hidden_bias,
            float* hidden_activations, int batch_size, int visible_size, int hidden_size) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * hidden_size) return;

            int batch_idx = idx / hidden_size;
            int hidden_idx = idx % hidden_size;

            float pre_activation = 0.0f;
            for (int v = 0; v < visible_size; ++v) {
                pre_activation += visible[batch_idx * visible_size + v] * weights[hidden_idx * visible_size + v];
            }

            if (hidden_bias != nullptr) {
                pre_activation += hidden_bias[hidden_idx];
            }

            hidden_activations[idx] = pre_activation;

        }

        __global__ void rbm_hidden_to_visible(const float* hidden_states, const float* weights, const float* visible_bias,
            float* visible_activations, int batch_size, int visible_size, int hidden_size) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * visible_size) return;

            int batch_idx = idx / visible_size;
            int visible_idx = idx % visible_size;

            float pre_activation = 0.0f;
            for (int h = 0; h < hidden_size; ++h) {
                pre_activation += hidden_states[batch_idx * hidden_size + h] * weights[h * visible_size + visible_idx];
            }

            if (visible_bias != nullptr) {
                pre_activation += visible_bias[visible_idx];
            }

            visible_activations[idx] = pre_activation;
        }

        __global__ void rbm_sample_states(const float* probs, float* states, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;

            // Simple thresholding at 0.5
            states[idx] = (probs[idx] > 0.5f) ? 1.0f : 0.0f;
        }

        __global__ void rbm_compute_gradients(const float* visible_data, const float* visible_recon,
            const float* hidden_probs, const float* hidden_recon_probs,
            float* grad_weights, float* grad_visible_bias, float* grad_hidden_bias,
            int batch_size, int visible_size, int hidden_size) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            // First handle weight gradients
            if (idx < hidden_size * visible_size) {
                int h = idx / visible_size;
                int v = idx % visible_size;

                float positive_phase = 0.0f;
                float negative_phase = 0.0f;

                for (int b = 0; b < batch_size; ++b) {
                    // Positive phase: visible_data * hidden_probs
                    positive_phase += visible_data[b * visible_size + v] * hidden_probs[b * hidden_size + h];

                    // Negative phase: visible_recon * hidden_recon_probs
                    negative_phase += visible_recon[b * visible_size + v] * hidden_recon_probs[b * hidden_size + h];
                }

                grad_weights[idx] = (positive_phase - negative_phase) / batch_size;
            }
            else if (idx < hidden_size * visible_size + visible_size) {
                int v = idx - (hidden_size * visible_size);

                float positive_phase = 0.0f;
                float negative_phase = 0.0f;

                for (int b = 0; b < batch_size; ++b) {
                    positive_phase += visible_data[b * visible_size + v];
                    negative_phase += visible_recon[b * visible_size + v];
                }

                grad_visible_bias[v] = (positive_phase - negative_phase) / batch_size;
            }
            else if (idx < hidden_size * visible_size + visible_size + hidden_size) {
                int h = idx - (hidden_size * visible_size + visible_size);

                float positive_phase = 0.0f;
                float negative_phase = 0.0f;

                for (int b = 0; b < batch_size; ++b) {
                    positive_phase += hidden_probs[b * hidden_size + h];
                    negative_phase += hidden_recon_probs[b * hidden_size + h];
                }

                // Update hidden bias gradient
                grad_hidden_bias[h] = (positive_phase - negative_phase) / batch_size;
            }
        }

        void launchRBMVisibleToHidden(const float* visible, const float* weights, const float* hidden_bias,
            float* hidden_activations, float* hidden_states, int batch_size, int visible_size, int hidden_size,
            cudaStream_t stream) {

            const int blockSize = 256;
            const int numElements = batch_size * hidden_size;
            const int numBlocks = (numElements + blockSize - 1) / blockSize;

            rbm_visible_to_hidden << <numBlocks, blockSize, 0, stream >> > (
                visible, weights, hidden_bias, hidden_activations,
                batch_size, visible_size, hidden_size
                );
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchRBMVisibleToHidden: %s\n", cudaGetErrorString(err));
            }
            cudaDeviceSynchronize();
            // Note: activation function and sampling will be applied in the RBMLayer class
        }

        void launchRBMHiddenToVisible(const float* hidden_states, const float* weights, const float* visible_bias,
            float* visible_activations, float* visible_recon, int batch_size, int visible_size, int hidden_size,
            cudaStream_t stream) {

            const int blockSize = 256;
            const int numElements = batch_size * visible_size;
            const int numBlocks = (numElements + blockSize - 1) / blockSize;

            rbm_hidden_to_visible << <numBlocks, blockSize, 0, stream >> > (
                hidden_states, weights, visible_bias, visible_activations,
                batch_size, visible_size, hidden_size
                );
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchRBMHiddenToVisible: %s\n", cudaGetErrorString(err));
            }
            cudaDeviceSynchronize();
        }

        void launchRBMSampleStates(const float* probs, float* states, int size, cudaStream_t stream) {
            const int blockSize = 256;
            const int numBlocks = (size + blockSize - 1) / blockSize;

            rbm_sample_states << <numBlocks, blockSize, 0, stream >> > (probs, states, size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchRBMSampleStates: %s\n", cudaGetErrorString(err));
            }
            cudaDeviceSynchronize();
        }

        void launchRBMComputeGradients(const float* visible_data, const float* visible_recon,
            const float* hidden_probs, const float* hidden_recon_probs,
            float* grad_weights, float* grad_visible_bias, float* grad_hidden_bias,
            int batch_size, int visible_size, int hidden_size,
            cudaStream_t stream) {

            const int blockSize = 256;
            // Total number of elements to process: weights + visible bias + hidden bias
            const int numElements = hidden_size * visible_size + visible_size + hidden_size;
            const int numBlocks = (numElements + blockSize - 1) / blockSize;

            rbm_compute_gradients << <numBlocks, blockSize, 0, stream >> > (
                visible_data, visible_recon, hidden_probs, hidden_recon_probs,
                grad_weights, grad_visible_bias, grad_hidden_bias,
                batch_size, visible_size, hidden_size
                );
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchRBMComputeGradients: %s\n", cudaGetErrorString(err));
            }
            cudaDeviceSynchronize();
        }
    }
}