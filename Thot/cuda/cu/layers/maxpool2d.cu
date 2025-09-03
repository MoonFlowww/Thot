#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdint.h>
#include <algorithm>
#include <cfloat>
#include "../../cuh/layers/maxpool2d.cuh"

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); } } while(0)
#endif

template <typename T>
__forceinline__ __device__ T ro(const T* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

namespace cuda::layers {

    __global__ void maxpool2d_forward( const float* __restrict__ input, float* __restrict__ output, int* __restrict__ max_indices, int batch_size, int channels, int in_height, int in_width, int kernel_size, int stride, int out_height, int out_width) {
        const int64_t total = (int64_t)batch_size * channels * out_height * out_width;
        for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < total;
             idx += (int64_t)blockDim.x * gridDim.x)
        {
            int64_t t = idx;
            const int w = t % out_width;      t /= out_width;
            const int h = t % out_height;     t /= out_height;
            const int c = t % channels;       t /= channels;
            const int b = t;
            const int h_start = h * stride;
            const int w_start = w * stride;
            float max_val = -FLT_MAX;
            int max_idx = -1;
            const int64_t base = ((int64_t)b * channels + c) * in_height;
            for (int i = 0; i < kernel_size; ++i) {
                const int ih = h_start + i;
                const int64_t rowBase = base + (int64_t)ih * in_width;
                for (int j = 0; j < kernel_size; ++j) {
                    const int iw = w_start + j;
                    const float val = ro(input + rowBase + iw);
                    if (val > max_val) {
                        max_val = val;
                        max_idx = (int)(rowBase + iw);
                    }
                }
            }
            output[idx] = max_val;
            if (max_indices) max_indices[idx] = max_idx;
        }
    }


    __global__ void maxpool2d_backward(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    const int* __restrict__ max_indices,
    int64_t total)
    {
        for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < total;
             idx += (int64_t)blockDim.x * gridDim.x)
        {
            const int max_idx = ro(max_indices + idx);
            if (max_idx >= 0) {
                atomicAdd(grad_input + max_idx, ro(grad_output + idx));
            }
        }
    }

    void launchMaxPool2DForward( const float* input, float* output, int* max_indices, int batch_size, int channels, int in_height, int in_width, int kernel_size, int stride, int out_height, int out_width, cudaStream_t stream) {
        const int64_t total = (int64_t)batch_size * channels * out_height * out_width;
        const int blockSize = 256;
        const int gridSize = (int)std::min<int64_t>((total + blockSize - 1) / blockSize, 65535);
        maxpool2d_forward<<<gridSize, blockSize, 0, stream>>>(
            input, output, max_indices,
            batch_size, channels, in_height, in_width,
            kernel_size, stride, out_height, out_width);
        CUDA_CHECK(cudaGetLastError());
    }

    void launchMaxPool2DBackward( const float* grad_output, float* grad_input, const int* max_indices, int batch_size, int channels, int in_height, int in_width, int kernel_size, int stride, int out_height, int out_width, cudaStream_t stream) {
        const int64_t total = (int64_t)batch_size * channels * out_height * out_width;
        const int blockSize = 256;
        const int gridSize = (int)std::min<int64_t>((total + blockSize - 1) / blockSize, 65535);
        maxpool2d_backward<<<gridSize, blockSize, 0, stream>>>(
            grad_output, grad_input, max_indices, total);
        CUDA_CHECK(cudaGetLastError());
    }


}
