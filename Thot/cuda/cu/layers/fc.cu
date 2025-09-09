#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cmath>
#include "../../cuh/layers/fc.cuh"


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


    __global__ void fc_forward(const float* __restrict__ input, const float* __restrict__ weights, const float* __restrict__ bias, float* __restrict__ output, int batch_size, int input_size, int output_size) {
        const int64_t N = (int64_t)batch_size * output_size;
        for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < N;
             idx += (int64_t)blockDim.x * gridDim.x)
        {
            int64_t t = idx;
            const int b = t / output_size;
            const int o = t - (int64_t)b * output_size;

            float sum = 0.0f;
            const int64_t inB = (int64_t)b * input_size;
            for (int i = 0; i < input_size; ++i) {
                sum += ro(input + inB + i) * ro(weights + (int64_t)i * output_size + o);
            }
            if (bias) sum += ro(bias + o);
            output[idx] = sum;
        }
    }

    __global__ void fc_backward_input(const float* __restrict__ grad_output, const float* __restrict__ weights, float* __restrict__ grad_input, int batch_size, int input_size, int output_size) {
        const int64_t N = (int64_t)batch_size * input_size;
        for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < N;
             idx += (int64_t)blockDim.x * gridDim.x)
        {
            int64_t t = idx;
            const int b = t / input_size;
            const int i = t - (int64_t)b * input_size;

            float sum = 0.0f;
            const int64_t goB = (int64_t)b * output_size;
            const int64_t wRow = (int64_t)i * output_size;
            for (int o = 0; o < output_size; ++o) {
                sum += ro(grad_output + goB + o) * ro(weights + wRow + o);
            }
            grad_input[idx] = sum;
        }
    }

    __global__ void fc_backward_weights(const float* __restrict__ input, const float* __restrict__ grad_output, float* __restrict__ grad_weights, int batch_size, int input_size, int output_size) {
        const int64_t N = (int64_t)input_size * output_size;
        for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < N;
             idx += (int64_t)blockDim.x * gridDim.x)
        {
            int64_t t = idx;
            const int i = t / output_size;
            const int o = t - (int64_t)i * output_size;

            float sum = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                const int64_t inIdx = (int64_t)b * input_size + i;
                const int64_t goIdx = (int64_t)b * output_size + o;
                sum += ro(input + inIdx) * ro(grad_output + goIdx);
            }
            grad_weights[idx] = sum / static_cast<float>(batch_size);
        }
        }

    __global__ void fc_backward_bias(const float* __restrict__ grad_output, float* __restrict__ grad_bias, int batch_size, int output_size) {
        const int oc = blockIdx.x * blockDim.x + threadIdx.x;
        if (oc >= output_size) return;

        float sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            sum += ro(grad_output + (int64_t)b * output_size + oc);
        }
        grad_bias[oc] = sum / static_cast<float>(batch_size);
    }

    void launchFCForward( const float* input, const float* weights, const float* bias, float* output, int batch_size, int input_size, int output_size, cudaStream_t stream) {
        const int64_t N = (int64_t)batch_size * output_size;
        const int blockSize = 256;
        const int gridSize = (int)std::min<int64_t>((N + blockSize - 1) / blockSize, 65535);
        fc_forward<<<gridSize, blockSize, 0, stream>>>(
            input, weights, bias, output,
            batch_size, input_size, output_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void launchFCBackwardInput(const float* grad_output, const float* weights, float* grad_input, int batch_size, int input_size, int output_size, cudaStream_t stream) {
        const int64_t N = (int64_t)batch_size * input_size;
        const int blockSize = 256;
        const int gridSize = (int)std::min<int64_t>((N + blockSize - 1) / blockSize, 65535);
        fc_backward_input<<<gridSize, blockSize, 0, stream>>>(
            grad_output, weights, grad_input,
            batch_size, input_size, output_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void launchFCBackwardWeights( const float* input, const float* grad_output, float* grad_weights, int batch_size, int input_size, int output_size, cudaStream_t stream) {
        const int64_t N = (int64_t)input_size * output_size;
        const int blockSize = 256;
        const int gridSize = (int)std::min<int64_t>((N + blockSize - 1) / blockSize, 65535);
        fc_backward_weights<<<gridSize, blockSize, 0, stream>>>(
            input, grad_output, grad_weights,
            batch_size, input_size, output_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void launchFCBackwardBias(const float* grad_output, float* grad_bias, int batch_size, int output_size, cudaStream_t stream) {
        const int blockSize = 256;
        const int gridSize = (output_size + blockSize - 1) / blockSize;
        fc_backward_bias<<<gridSize, blockSize, 0, stream>>>(
            grad_output, grad_bias,
            batch_size, output_size);
        CUDA_CHECK(cudaGetLastError());
    }

} // namespace cuda::layers
