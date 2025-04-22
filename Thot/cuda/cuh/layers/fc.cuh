#pragma once

#include <cuda_runtime.h>

namespace cuda {
    namespace layers {
        // Forward pass kernel
        __global__ void fc_forward(const float* input, const float* weights, const float* bias,
            float* output, int batch_size, int input_size, int output_size);

        // Backward pass kernels
        __global__ void fc_backward_input(const float* grad_output, const float* weights,
            float* grad_input, int batch_size, int input_size, int output_size);

        __global__ void fc_backward_weights(const float* input, const float* grad_output,
            float* grad_weights, int batch_size, int input_size, int output_size);

        __global__ void fc_backward_bias(const float* grad_output, float* grad_bias,
            int batch_size, int output_size);



        // Launch functions for kernels
        void launchFCForward(const float* input, const float* weights, const float* bias,
            float* output, int batch_size, int input_size, int output_size,
            cudaStream_t stream = 0);

        void launchFCBackwardInput(const float* grad_output, const float* weights,
            float* grad_input, int batch_size, int input_size, int output_size,
            cudaStream_t stream = 0);

        void launchFCBackwardWeights(const float* input, const float* grad_output,
            float* grad_weights, int batch_size, int input_size, int output_size,
            cudaStream_t stream = 0);

        void launchFCBackwardBias(const float* grad_output, float* grad_bias,
            int batch_size, int output_size,
            cudaStream_t stream = 0);


    }
}