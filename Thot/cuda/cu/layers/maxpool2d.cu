#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "../../cuh/layers/maxpool2d.cuh"

#include <c++/15.2.1/cfloat>

namespace cuda {
    namespace layers {

        __global__ void maxpool2d_forward(const float* input, float* output, int* max_indices,
            int batch_size, int channels, int in_height, int in_width,
            int kernel_size, int stride, int out_height, int out_width) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total = batch_size * channels * out_height * out_width;
            if (idx >= total) return;

            int w = idx % out_width;
            int h = (idx / out_width) % out_height;
            int c = (idx / (out_width * out_height)) % channels;
            int b = idx / (out_width * out_height * channels);

            int h_start = h * stride;
            int w_start = w * stride;

            float max_val = -FLT_MAX;
            int max_idx = -1;
            for (int i = 0; i < kernel_size; ++i) {
                for (int j = 0; j < kernel_size; ++j) {
                    int ih = h_start + i;
                    int iw = w_start + j;
                    int input_idx = ((b * channels + c) * in_height + ih) * in_width + iw;
                    float val = input[input_idx];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = input_idx;
                    }
                }
            }
            output[idx] = max_val;
            if (max_indices) max_indices[idx] = max_idx;
        }

        __global__ void maxpool2d_backward(const float* grad_output, float* grad_input, const int* max_indices, int total) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total) return;
            int max_idx = max_indices[idx];
            if (max_idx >= 0) {
                atomicAdd(&grad_input[max_idx], grad_output[idx]);
            }
        }

        void launchMaxPool2DForward(const float* input, float* output, int* max_indices,
            int batch_size, int channels, int in_height, int in_width,
            int kernel_size, int stride, int out_height, int out_width, cudaStream_t stream) {
            int total = batch_size * channels * out_height * out_width;
            int blockSize = 256;
            int numBlocks = (total + blockSize - 1) / blockSize;
            maxpool2d_forward<<<numBlocks, blockSize, 0, stream>>>(input, output, max_indices,
                batch_size, channels, in_height, in_width, kernel_size, stride, out_height, out_width);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchMaxPool2DForward: %s\n", cudaGetErrorString(err));
            }
            cudaDeviceSynchronize();
        }

        void launchMaxPool2DBackward(const float* grad_output, float* grad_input, const int* max_indices,
            int batch_size, int channels, int in_height, int in_width,
            int kernel_size, int stride, int out_height, int out_width, cudaStream_t stream) {
            int total = batch_size * channels * out_height * out_width;
            int blockSize = 256;
            int numBlocks = (total + blockSize - 1) / blockSize;
            maxpool2d_backward<<<numBlocks, blockSize, 0, stream>>>(grad_output, grad_input, max_indices, total);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchMaxPool2DBackward: %s\n", cudaGetErrorString(err));
            }
            cudaDeviceSynchronize();
        }

    }
}
