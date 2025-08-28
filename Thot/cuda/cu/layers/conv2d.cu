#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cmath>
#include "../../cuh/layers/conv2d.cuh"

namespace cuda {
    namespace layers {

        __global__ void conv2d_forward(const float* input, const float* weights, const float* bias,
            float* output, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * out_channels * out_height * out_width) return;

            int b = idx / (out_channels * out_height * out_width);
            int oc = (idx / (out_height * out_width)) % out_channels;
            int h = (idx / out_width) % out_height;
            int w = idx % out_width;

            float sum = 0.0f;

            for (int ic = 0; ic < in_channels; ++ic) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int ih = h * stride - padding + kh;
                        int iw = w * stride - padding + kw;

                        if (ih < 0 || ih >= in_height || iw < 0 || iw >= in_width) continue;

                        // Input index
                        int in_idx = b * (in_channels * in_height * in_width) +
                            ic * (in_height * in_width) +
                            ih * in_width +
                            iw;

                        // Weight index
                        int w_idx = oc * (in_channels * kernel_size * kernel_size) +
                            ic * (kernel_size * kernel_size) +
                            kh * kernel_size +
                            kw;

                        sum += input[in_idx] * weights[w_idx];
                    }
                }
            }

            if (bias != nullptr) sum += bias[oc];
            

            output[idx] = sum;
        }

        __global__ void conv2d_backward_input(const float* grad_output, const float* weights,
            float* grad_input, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * in_channels * in_height * in_width) return;

            int b = idx / (in_channels * in_height * in_width);
            int ic = (idx / (in_height * in_width)) % in_channels;
            int h = (idx / in_width) % in_height;
            int w = idx % in_width;

            float sum = 0.0f;

            for (int oc = 0; oc < out_channels; ++oc) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        // Calculate output position
                        int oh = (h + padding - kh) / stride;
                        int ow = (w + padding - kw) / stride;

                        if (oh >= 0 && oh < out_height && ow >= 0 && ow < out_width &&
                            (h + padding - kh) % stride == 0 && (w + padding - kw) % stride == 0) {

                            // Grad output index
                            int go_idx = b * (out_channels * out_height * out_width) +
                                oc * (out_height * out_width) +
                                oh * out_width +
                                ow;

                            // Weight index (reversed for backward)
                            int w_idx = oc * (in_channels * kernel_size * kernel_size) +
                                ic * (kernel_size * kernel_size) +
                                kh * kernel_size +
                                kw;

                            sum += grad_output[go_idx] * weights[w_idx];
                        }
                    }
                }
            }

            grad_input[idx] = sum;
        }

        __global__ void conv2d_backward_weights(const float* input, const float* grad_output,
            float* grad_weights, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= out_channels * in_channels * kernel_size * kernel_size) return;

            int oc = idx / (in_channels * kernel_size * kernel_size);
            int ic = (idx / (kernel_size * kernel_size)) % in_channels;
            int kh = (idx / kernel_size) % kernel_size;
            int kw = idx % kernel_size;

            float sum = 0.0f;

            for (int b = 0; b < batch_size; ++b) {
                for (int oh = 0; oh < out_height; ++oh) {
                    for (int ow = 0; ow < out_width; ++ow) {
                        int ih = oh * stride - padding + kh;
                        int iw = ow * stride - padding + kw;

                        if (ih < 0 || ih >= in_height || iw < 0 || iw >= in_width) continue;

                        int in_idx = b * (in_channels * in_height * in_width) +
                            ic * (in_height * in_width) +
                            ih * in_width +
                            iw;

                        int go_idx = b * (out_channels * out_height * out_width) +
                            oc * (out_height * out_width) +
                            oh * out_width +
                            ow;

                        sum += input[in_idx] * grad_output[go_idx];
                    }
                }
            }

            grad_weights[idx] = sum;
        }

        __global__ void conv2d_backward_bias(const float* grad_output, float* grad_bias,
            int batch_size, int out_channels, int out_height, int out_width) {

            int oc = blockIdx.x * blockDim.x + threadIdx.x;
            if (oc >= out_channels) return;

            float sum = 0.0f;

            for (int b = 0; b < batch_size; ++b) {
                for (int oh = 0; oh < out_height; ++oh) {
                    for (int ow = 0; ow < out_width; ++ow) {
                        int go_idx = b * (out_channels * out_height * out_width) +
                            oc * (out_height * out_width) +
                            oh * out_width +
                            ow;

                        sum += grad_output[go_idx];
                    }
                }
            }

            grad_bias[oc] = sum;
        }

        void launchConv2DForward(const float* input, const float* weights, const float* bias,
            float* output, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
            cudaStream_t stream) {

            const int blockSize = 256;
            const int numElements = batch_size * out_channels * out_height * out_width;
            const int numBlocks = (numElements + blockSize - 1) / blockSize;

            conv2d_forward << <numBlocks, blockSize, 0, stream >> > (
                input, weights, bias, output,
                batch_size, in_channels, in_height, in_width,
                out_channels, kernel_size, stride, padding,
                out_height, out_width
                );

            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("CUDA error after Conv2D forward: %s\n", cudaGetErrorString(err));
            }
        }

        void launchConv2DBackwardInput(const float* grad_output, const float* weights,
            float* grad_input, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
            cudaStream_t stream) {

            const int blockSize = 256;
            const int numElements = batch_size * in_channels * in_height * in_width;
            const int numBlocks = (numElements + blockSize - 1) / blockSize;

            conv2d_backward_input << <numBlocks, blockSize, 0, stream >> > (
                grad_output, weights, grad_input,
                batch_size, in_channels, in_height, in_width,
                out_channels, kernel_size, stride, padding,
                out_height, out_width
                );
        }

        void launchConv2DBackwardWeights(const float* input, const float* grad_output,
            float* grad_weights, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
            cudaStream_t stream) {

            const int blockSize = 256;
            const int numElements = out_channels * in_channels * kernel_size * kernel_size;
            const int numBlocks = (numElements + blockSize - 1) / blockSize;

            conv2d_backward_weights << <numBlocks, blockSize, 0, stream >> > (
                input, grad_output, grad_weights,
                batch_size, in_channels, in_height, in_width,
                out_channels, kernel_size, stride, padding,
                out_height, out_width
                );
        }

        void launchConv2DBackwardBias(const float* grad_output, float* grad_bias,
            int batch_size, int out_channels, int out_height, int out_width,
            cudaStream_t stream) {

            const int blockSize = 256;
            const int numBlocks = (out_channels + blockSize - 1) / blockSize;

            conv2d_backward_bias << <numBlocks, blockSize, 0, stream >> > (
                grad_output, grad_bias,
                batch_size, out_channels, out_height, out_width
                );
        }
    }
}