#pragma once

#include <cuda_runtime.h>

namespace cuda {
    namespace layers {
        // Forward pass kernels
        __global__ void rbm_visible_to_hidden(const float* visible, const float* weights, const float* hidden_bias,
            float* hidden_activations, int batch_size, int visible_size, int hidden_size);

        __global__ void rbm_hidden_to_visible(const float* hidden_states, const float* weights, const float* visible_bias,
            float* visible_activations, int batch_size, int visible_size, int hidden_size);

        __global__ void rbm_sample_states(const float* probs, float* states, int size);

        __global__ void rbm_compute_gradients(const float* visible_data, const float* visible_recon,
            const float* hidden_probs, const float* hidden_recon_probs,
            float* grad_weights, float* grad_visible_bias, float* grad_hidden_bias,
            int batch_size, int visible_size, int hidden_size);

        void launchRBMVisibleToHidden(const float* visible, const float* weights, const float* hidden_bias,
            float* hidden_activations, float* hidden_states, int batch_size, int visible_size, int hidden_size,
            cudaStream_t stream = 0);

        void launchRBMHiddenToVisible(const float* hidden_states, const float* weights, const float* visible_bias,
            float* visible_activations, float* visible_recon, int batch_size, int visible_size, int hidden_size,
            cudaStream_t stream = 0);

        void launchRBMSampleStates(const float* probs, float* states, int size, cudaStream_t stream = 0);

        void launchRBMComputeGradients(const float* visible_data, const float* visible_recon,
            const float* hidden_probs, const float* hidden_recon_probs,
            float* grad_weights, float* grad_visible_bias, float* grad_hidden_bias,
            int batch_size, int visible_size, int hidden_size,
            cudaStream_t stream = 0);
    }
}