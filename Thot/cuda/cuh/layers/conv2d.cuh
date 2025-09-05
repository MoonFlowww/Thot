#pragma once

#include <cuda_runtime.h>

namespace cuda {
    namespace layers {
        enum class ConvAlgo { Auto, Direct, Winograd, FFT }; // latency optim techs

        __global__ void conv2d_forward(const float* input, const float* weights, const float* bias,
            float* output, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width);

        // Backward pass kernels
        __global__ void conv2d_backward_input(const float* grad_output, const float* weights,
            float* grad_input, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width);

        __global__ void conv2d_backward_weights(const float* input, const float* grad_output,
            float* grad_weights, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width);

        __global__ void conv2d_backward_bias(const float* grad_output, float* grad_bias,
            int batch_size, int out_channels, int out_height, int out_width);

        void launchConv2DForward(const float* input, const float* weights, const float* bias,
            float* output, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
            ConvAlgo algo = ConvAlgo::Auto,
            cudaStream_t stream = 0);

        void launchConv2DBackwardInput(const float* grad_output, const float* weights,
            float* grad_input, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
            ConvAlgo algo = ConvAlgo::Auto,
            cudaStream_t stream = 0);

        void launchConv2DBackwardWeights(const float* input, const float* grad_output,
            float* grad_weights, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
            ConvAlgo algo = ConvAlgo::Auto,
            cudaStream_t stream = 0);

        void launchConv2DBackwardBias(const float* grad_output, float* grad_bias,
            int batch_size, int out_channels, int out_height, int out_width,
            cudaStream_t stream = 0);
    }
}