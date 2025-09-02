#pragma once

#include <cuda_runtime.h>

namespace cuda {
    namespace layers {
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

        // Utility kernel to add bias to the output matrix
        __global__ void add_bias(float* C, const float* bias, int M, int N);

        void launchConv2DForward(const float* input, const float* weights, const float* bias,
            float* output, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
            cudaStream_t stream = 0);

        // GEMM-based implementation using im2col transformation
        void launchConv2DForwardIm2Col(const float* input, const float* weights, const float* bias,
            float* output, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
            cudaStream_t stream = 0);


        void launchConv2DBackwardInput(const float* grad_output, const float* weights,
            float* grad_input, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
            cudaStream_t stream = 0);

        void launchConv2DBackwardWeights(const float* input, const float* grad_output,
            float* grad_weights, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
            cudaStream_t stream = 0);

        void launchConv2DBackwardBias(const float* grad_output, float* grad_bias,
            int batch_size, int out_channels, int out_height, int out_width,
            cudaStream_t stream = 0);

#ifdef THOT_WITH_CUDNN
        // cuDNN accelerated variants
        void launchConv2DForwardCuDNN(const float* input, const float* weights, const float* bias,
            float* output, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
            cudaStream_t stream = 0);

        void launchConv2DBackwardInputCuDNN(const float* grad_output, const float* weights,
            float* grad_input, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
            cudaStream_t stream = 0);

        void launchConv2DBackwardWeightsCuDNN(const float* input, const float* grad_output,
            float* grad_weights, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
            cudaStream_t stream = 0);

        void launchConv2DBackwardBiasCuDNN(const float* grad_output, float* grad_bias,
            int batch_size, int out_channels, int out_height, int out_width,
            cudaStream_t stream = 0);
#endif
    }
}