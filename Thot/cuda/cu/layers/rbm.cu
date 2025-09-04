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

#define TILE_DIM 32

template <typename T>
__forceinline__ __device__ T ro(const T* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

namespace cuda::layers {

// Optimized forward: H = V * W^T + b
__global__ void rbm_visible_to_hidden(
    const float* __restrict__ visible,
    const float* __restrict__ weights,
    const float* __restrict__ hidden_bias,
    float* __restrict__ hidden_activations,
    int batch_size, int visible_size, int hidden_size)
{
    __shared__ float sV[TILE_DIM][TILE_DIM];
    __shared__ float sW[TILE_DIM][TILE_DIM];

    int b = blockIdx.y * TILE_DIM + threadIdx.y;
    int h = blockIdx.x * TILE_DIM + threadIdx.x;

    if (b >= batch_size || h >= hidden_size) return;

    float sum = 0.0f;
    for (int tile = 0; tile < (visible_size + TILE_DIM - 1) / TILE_DIM; tile++) {
        int v = tile * TILE_DIM + threadIdx.x;
        int v2 = tile * TILE_DIM + threadIdx.y;

        if (v < visible_size && b < batch_size)
            sV[threadIdx.y][threadIdx.x] = visible[b * visible_size + v];
        else
            sV[threadIdx.y][threadIdx.x] = 0.0f;

        if (v2 < visible_size && h < hidden_size)
            sW[threadIdx.y][threadIdx.x] = weights[h * visible_size + v2];
        else
            sW[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_DIM; k++) {
            sum += sV[threadIdx.y][k] * sW[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (hidden_bias) sum += hidden_bias[h];
    hidden_activations[b * hidden_size + h] = sum;
}

// Optimized backward: V = H * W + b
__global__ void rbm_hidden_to_visible(
    const float* __restrict__ hidden,
    const float* __restrict__ weights,
    const float* __restrict__ visible_bias,
    float* __restrict__ visible_activations,
    int batch_size, int visible_size, int hidden_size)
{
    __shared__ float sH[TILE_DIM][TILE_DIM];
    __shared__ float sW[TILE_DIM][TILE_DIM];

    int b = blockIdx.y * TILE_DIM + threadIdx.y;
    int v = blockIdx.x * TILE_DIM + threadIdx.x;

    if (b >= batch_size || v >= visible_size) return;

    float sum = 0.0f;
    for (int tile = 0; tile < (hidden_size + TILE_DIM - 1) / TILE_DIM; tile++) {
        int h = tile * TILE_DIM + threadIdx.x;
        int h2 = tile * TILE_DIM + threadIdx.y;

        if (h < hidden_size && b < batch_size)
            sH[threadIdx.y][threadIdx.x] = hidden[b * hidden_size + h];
        else
            sH[threadIdx.y][threadIdx.x] = 0.0f;

        if (h2 < hidden_size && v < visible_size)
            sW[threadIdx.y][threadIdx.x] = weights[h2 * visible_size + v];
        else
            sW[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_DIM; k++) {
            sum += sH[threadIdx.y][k] * sW[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (visible_bias) sum += visible_bias[v];
    visible_activations[b * visible_size + v] = sum;
}

// Gradient computation with GEMM-like logic
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
    __shared__ float sV[TILE_DIM][TILE_DIM];
    __shared__ float sH[TILE_DIM][TILE_DIM];

    int h = blockIdx.y * TILE_DIM + threadIdx.y;
    int v = blockIdx.x * TILE_DIM + threadIdx.x;

    if (h < hidden_size && v < visible_size) {
        float pos = 0.0f, neg = 0.0f;
        for (int tile = 0; tile < (batch_size + TILE_DIM - 1) / TILE_DIM; tile++) {
            int bV = tile * TILE_DIM + threadIdx.y;
            int bH = tile * TILE_DIM + threadIdx.x;

            if (bV < batch_size && v < visible_size)
                sV[threadIdx.y][threadIdx.x] = visible_data[bV * visible_size + v];
            else
                sV[threadIdx.y][threadIdx.x] = 0.0f;

            if (bH < batch_size && h < hidden_size)
                sH[threadIdx.y][threadIdx.x] = hidden_probs[bH * hidden_size + h];
            else
                sH[threadIdx.y][threadIdx.x] = 0.0f;

            __syncthreads();
            for (int k = 0; k < TILE_DIM; k++)
                pos += sV[k][threadIdx.x] * sH[threadIdx.y][k];
            __syncthreads();

            if (bV < batch_size && v < visible_size)
                sV[threadIdx.y][threadIdx.x] = visible_recon[bV * visible_size + v];
            else
                sV[threadIdx.y][threadIdx.x] = 0.0f;

            if (bH < batch_size && h < hidden_size)
                sH[threadIdx.y][threadIdx.x] = hidden_recon_probs[bH * hidden_size + h];
            else
                sH[threadIdx.y][threadIdx.x] = 0.0f;

            __syncthreads();
            for (int k = 0; k < TILE_DIM; k++)
                neg += sV[k][threadIdx.x] * sH[threadIdx.y][k];
            __syncthreads();
        }
        grad_weights[h * visible_size + v] = (pos - neg) / batch_size;
    }

    // Bias gradients (reduction over batch)
    if (h == 0 && v < visible_size) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++)
            sum += visible_data[b * visible_size + v] - visible_recon[b * visible_size + v];
        grad_visible_bias[v] = sum / batch_size;
    }
    if (v == 0 && h < hidden_size) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++)
            sum += hidden_probs[b * hidden_size + h] - hidden_recon_probs[b * hidden_size + h];
        grad_hidden_bias[h] = sum / batch_size;
    }
}

// Launchers
void launchRBMVisibleToHidden(const float* visible, const float* weights, const float* hidden_bias, float* hidden_activations, float* hidden_states, int batch_size, int visible_size, int hidden_size, cudaStream_t stream) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((hidden_size + TILE_DIM - 1) / TILE_DIM, (batch_size + TILE_DIM - 1) / TILE_DIM);
    rbm_visible_to_hidden<<<grid, block, 0, stream>>>(visible, weights, hidden_bias, hidden_activations, batch_size, visible_size, hidden_size);
    CUDA_CHECK(cudaGetLastError());
}

void launchRBMHiddenToVisible(const float* hidden_states, const float* weights, const float* visible_bias, float* visible_activations, float* visible_recon, int batch_size, int visible_size, int hidden_size, cudaStream_t stream) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((visible_size + TILE_DIM - 1) / TILE_DIM, (batch_size + TILE_DIM - 1) / TILE_DIM);
    rbm_hidden_to_visible<<<grid, block, 0, stream>>>(hidden_states, weights, visible_bias, visible_activations, batch_size, visible_size, hidden_size);
    CUDA_CHECK(cudaGetLastError());
}

void launchRBMComputeGradients(const float* visible_data, const float* visible_recon, const float* hidden_probs, const float* hidden_recon_probs, float* grad_weights, float* grad_visible_bias, float* grad_hidden_bias, int batch_size, int visible_size, int hidden_size, cudaStream_t stream) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((visible_size + TILE_DIM - 1) / TILE_DIM, (hidden_size + TILE_DIM - 1) / TILE_DIM);
    rbm_compute_gradients<<<grid, block, 0, stream>>>(visible_data, visible_recon, hidden_probs, hidden_recon_probs, grad_weights, grad_visible_bias, grad_hidden_bias, batch_size, visible_size, hidden_size);
    CUDA_CHECK(cudaGetLastError());
}

}
