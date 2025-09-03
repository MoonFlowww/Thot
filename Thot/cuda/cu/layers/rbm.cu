#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <curand_kernel.h>
#include "../../cuh/layers/rbm.cuh"


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

    __global__ void rbm_visible_to_hidden(
    const float* __restrict__ visible,
    const float* __restrict__ weights,
    const float* __restrict__ hidden_bias,
    float* __restrict__ hidden_activations,
    int batch_size, int visible_size, int hidden_size)
    {
        const int64_t N = (int64_t)batch_size * hidden_size;
        for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < N;
             idx += (int64_t)blockDim.x * gridDim.x)
        {
            int64_t t = idx;
            const int b = t / hidden_size;
            const int h = t - (int64_t)b * hidden_size;

            float pre_activation = 0.0f;
            const int64_t visB = (int64_t)b * visible_size;
            const int64_t wRow = (int64_t)h * visible_size;
            for (int v = 0; v < visible_size; ++v) {
                pre_activation += ro(visible + visB + v) * ro(weights + wRow + v);
            }
            if (hidden_bias) pre_activation += ro(hidden_bias + h);
            hidden_activations[idx] = pre_activation;
        }
    }

    __global__ void rbm_hidden_to_visible(
        const float* __restrict__ hidden_states,
        const float* __restrict__ weights,
        const float* __restrict__ visible_bias,
        float* __restrict__ visible_activations,
        int batch_size, int visible_size, int hidden_size)
    {
        const int64_t N = (int64_t)batch_size * visible_size;
        for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < N;
             idx += (int64_t)blockDim.x * gridDim.x)
        {
            int64_t t = idx;
            const int b = t / visible_size;
            const int v = t - (int64_t)b * visible_size;

            float pre_activation = 0.0f;
            const int64_t hidB = (int64_t)b * hidden_size;
            for (int h = 0; h < hidden_size; ++h) {
                pre_activation += ro(hidden_states + hidB + h) * ro(weights + (int64_t)h * visible_size + v);
            }
            if (visible_bias) pre_activation += ro(visible_bias + v);
            visible_activations[idx] = pre_activation;
        }
    }

    __global__ void rbm_sample_states(
        const float* __restrict__ probs,
        float* __restrict__ states,
        int size)
    {
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < size;
             idx += blockDim.x * gridDim.x)
        {
            states[idx] = (ro(probs + idx) > 0.5f) ? 1.0f : 0.0f;
        }
    }

    __global__ void rbm_compute_gradients(
        const float* __restrict__ visible_data,
        const float* __restrict__ visible_recon,
        const float* __restrict__ hidden_probs,
        const float* __restrict__ hidden_recon_probs,
        float* __restrict__ grad_weights,
        float* __restrict__ grad_visible_bias,
        float* __restrict__ grad_hidden_bias,
        int batch_size, int visible_size, int hidden_size)
    {
        const int64_t weightsCount = (int64_t)hidden_size * visible_size;
        const int64_t vbiasOffset = weightsCount;
        const int64_t hbiasOffset = vbiasOffset + visible_size;
        const int64_t total = hbiasOffset + hidden_size;

        for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < total;
             idx += (int64_t)blockDim.x * gridDim.x)
        {
            if (idx < weightsCount) {
                const int h = idx / visible_size;
                const int v = idx - (int64_t)h * visible_size;

                float pos = 0.0f, neg = 0.0f;
                for (int b = 0; b < batch_size; ++b) {
                    const int64_t vIdx = (int64_t)b * visible_size + v;
                    const int64_t hIdx = (int64_t)b * hidden_size + h;
                    pos += ro(visible_data + vIdx) * ro(hidden_probs + hIdx);
                    neg += ro(visible_recon + vIdx) * ro(hidden_recon_probs + hIdx);
                }
                grad_weights[idx] = (pos - neg) / static_cast<float>(batch_size);
            }
            else if (idx < hbiasOffset) {
                const int v = idx - vbiasOffset;
                float pos = 0.0f, neg = 0.0f;
                for (int b = 0; b < batch_size; ++b) {
                    const int64_t vIdx = (int64_t)b * visible_size + v;
                    pos += ro(visible_data + vIdx);
                    neg += ro(visible_recon + vIdx);
                }
                grad_visible_bias[v] = (pos - neg) / static_cast<float>(batch_size);
            }
            else {
                const int h = idx - hbiasOffset;
                float pos = 0.0f, neg = 0.0f;
                for (int b = 0; b < batch_size; ++b) {
                    const int64_t hIdx = (int64_t)b * hidden_size + h;
                    pos += ro(hidden_probs + hIdx);
                    neg += ro(hidden_recon_probs + hIdx);
                }
                grad_hidden_bias[h] = (pos - neg) / static_cast<float>(batch_size);
            }
        }
        }

    void launchRBMVisibleToHidden(const float* visible, const float* weights, const float* hidden_bias, float* hidden_activations, float* hidden_states, int batch_size, int visible_size, int hidden_size, cudaStream_t stream) {
        const int64_t numElements = (int64_t)batch_size * hidden_size;
        const int blockSize = 256;
        const int gridSize = (int)std::min<int64_t>((numElements + blockSize - 1) / blockSize, 65535);
        rbm_visible_to_hidden<<<gridSize, blockSize, 0, stream>>>(
            visible, weights, hidden_bias, hidden_activations,
            batch_size, visible_size, hidden_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void launchRBMHiddenToVisible(const float* hidden_states, const float* weights, const float* visible_bias, float* visible_activations, float* visible_recon, int batch_size, int visible_size, int hidden_size, cudaStream_t stream) {
        const int64_t numElements = (int64_t)batch_size * visible_size;
        const int blockSize = 256;
        const int gridSize = (int)std::min<int64_t>((numElements + blockSize - 1) / blockSize, 65535);
        rbm_hidden_to_visible<<<gridSize, blockSize, 0, stream>>>(
            hidden_states, weights, visible_bias, visible_activations,
            batch_size, visible_size, hidden_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void launchRBMSampleStates(const float* probs, float* states, int size, cudaStream_t stream)
    {
        const int blockSize = 256;
        const int gridSize = (size + blockSize - 1) / blockSize;
        rbm_sample_states<<<gridSize, blockSize, 0, stream>>>(probs, states, size);
        CUDA_CHECK(cudaGetLastError());
    }

    void launchRBMComputeGradients(const float* visible_data, const float* visible_recon, const float* hidden_probs, const float* hidden_recon_probs,float* grad_weights, float* grad_visible_bias, float* grad_hidden_bias, int batch_size, int visible_size, int hidden_size, cudaStream_t stream) {
        const int64_t weightsCount = (int64_t)hidden_size * visible_size;
        const int64_t total = weightsCount + visible_size + hidden_size;
        const int blockSize = 256;
        const int gridSize = (int)std::min<int64_t>((total + blockSize - 1) / blockSize, 65535);
        rbm_compute_gradients<<<gridSize, blockSize, 0, stream>>>(
            visible_data, visible_recon,
            hidden_probs, hidden_recon_probs,
            grad_weights, grad_visible_bias, grad_hidden_bias,
            batch_size, visible_size, hidden_size);
        CUDA_CHECK(cudaGetLastError());
    }

}