#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include "../../cuh/layers/conv2d.cuh"

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); } } while(0)
#endif

namespace cuda {
namespace layers {

__forceinline__ __device__ int div_floor(int a, int b) {
    // b > 0
    int q = a / b;
    int r = a % b;
    if ((r != 0) && ((r > 0) != (b > 0))) --q;
    return q;
}
__forceinline__ __device__ int div_ceil(int a, int b) {
    // b > 0
    return -div_floor(-a, b);
}

template <typename T>
__forceinline__ __device__ T ro(const T* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__global__ void conv2d_forward(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding,
    int out_height, int out_width)
{
    const int64_t outHW = (int64_t)out_height * out_width;
    const int64_t inHW  = (int64_t)in_height * in_width;
    const int64_t perB  = (int64_t)out_channels * outHW;
    const int64_t N     = (int64_t)batch_size * perB;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N;
         idx += (int64_t)blockDim.x * gridDim.x)
    {
        int64_t t = idx;
        const int b  = t / perB;           t -= (int64_t)b * perB;
        const int oc = t / outHW;          t -= (int64_t)oc * outHW;
        const int h  = t / out_width;
        const int w  = t - h * out_width;

        float sum = 0.0f;

        // anchor in input
        const int ih0 = h * stride - padding;
        const int iw0 = w * stride - padding;

        // valid filter ranges for borders (case-independent)
        const int kh_min = max(0, -ih0);
        const int kw_min = max(0, -iw0);
        const int kh_max = min(kernel_size, in_height - ih0);
        const int kw_max = min(kernel_size, in_width  - iw0);

        const int64_t inB  = (int64_t)b  * in_channels * inHW;
        const int64_t wOC  = (int64_t)oc * in_channels * kernel_size * kernel_size;

        for (int ic = 0; ic < in_channels; ++ic) {
            const int64_t inC = inB + (int64_t)ic * inHW;
            const int64_t wIC = wOC + (int64_t)ic * kernel_size * kernel_size;

            for (int kh = kh_min; kh < kh_max; ++kh) {
                const int ih = ih0 + kh;
                const int64_t inRow = inC + (int64_t)ih * in_width;

                const int64_t wRow = wIC + (int64_t)kh * kernel_size;
                for (int kw = kw_min; kw < kw_max; ++kw) {
                    const int iw = iw0 + kw;
                    sum += ro(input + inRow + iw) * ro(weights + wRow + kw);
                }
            }
        }

        if (bias) sum += ro(bias + oc);
        output[idx] = sum;
    }
}

__global__ void conv2d_backward_input(
    const float* __restrict__ grad_output,
    const float* __restrict__ weights,
    float* __restrict__ grad_input,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding,
    int out_height, int out_width)
{
    const int64_t inHW  = (int64_t)in_height * in_width;
    const int64_t outHW = (int64_t)out_height * out_width;
    const int64_t perB  = (int64_t)in_channels * inHW;
    const int64_t N     = (int64_t)batch_size * perB;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N;
         idx += (int64_t)blockDim.x * gridDim.x)
    {
        int64_t t = idx;
        const int b  = t / perB;          t -= (int64_t)b * perB;
        const int ic = t / inHW;          t -= (int64_t)ic * inHW;
        const int h  = t / in_width;
        const int w  = t - h * in_width;

        float sum = 0.0f;

        // oh, ow ranges that hit this (h,w)
        // ih = oh*stride - padding + kh  in [0, in_h-1] and 0<=kh<K => oh in:
        const int oh_start = max(0, div_ceil(h + padding - (kernel_size - 1), stride));
        const int oh_end   = min(out_height - 1, div_floor(h + padding, stride));
        const int ow_start = max(0, div_ceil(w + padding - (kernel_size - 1), stride));
        const int ow_end   = min(out_width  - 1, div_floor(w + padding, stride));

        const int64_t goB = (int64_t)b * out_channels * outHW;

        for (int oc = 0; oc < out_channels; ++oc) {
            const int64_t goOC = goB + (int64_t)oc * outHW;
            const int64_t wOC  = (int64_t)oc * in_channels * kernel_size * kernel_size
                               + (int64_t)ic * kernel_size * kernel_size;

            for (int oh = oh_start; oh <= oh_end; ++oh) {
                const int kh = h + padding - oh * stride; // in [0, K-1]
                const int64_t goRow = goOC + (int64_t)oh * out_width;

                const int64_t wRow  = wOC + (int64_t)kh * kernel_size;

                for (int ow = ow_start; ow <= ow_end; ++ow) {
                    const int kw = w + padding - ow * stride; // in [0, K-1]
                    sum += ro(grad_output + goRow + ow) * ro(weights + wRow + kw);
                }
            }
        }

        grad_input[idx] = sum;
    }
}

__global__ void conv2d_backward_weights(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_weights,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding,
    int out_height, int out_width)
{
    const int64_t K2     = (int64_t)kernel_size * kernel_size;
    const int64_t Nw     = (int64_t)out_channels * in_channels * K2;
    const int64_t inHW   = (int64_t)in_height * in_width;
    const int64_t outHW  = (int64_t)out_height * out_width;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < Nw;
         idx += (int64_t)blockDim.x * gridDim.x)
    {
        int64_t t = idx;
        const int oc = t / (in_channels * K2); t -= (int64_t)oc * (in_channels * K2);
        const int ic = t / K2;                 t -= (int64_t)ic * K2;
        const int kh = t / kernel_size;
        const int kw = t - kh * kernel_size;

        float sum = 0.0f;

        // oh, ow ranges that make ih,iw in-bounds
        // ih = oh*stride - padding + kh in [0, in_h-1]
        const int oh_start = max(0, div_ceil(        padding - kh, stride));
        const int oh_end   = min(out_height - 1, div_floor(in_height - 1 + padding - kh, stride));
        // iw = ow*stride - padding + kw in [0, in_w-1]
        const int ow_start = max(0, div_ceil(        padding - kw, stride));
        const int ow_end   = min(out_width  - 1, div_floor(in_width  - 1 + padding - kw, stride));

        for (int b = 0; b < batch_size; ++b) {
            const int64_t inB = (int64_t)b * in_channels * inHW + (int64_t)ic * inHW;
            const int64_t goB = (int64_t)b * out_channels * outHW + (int64_t)oc * outHW;

            for (int oh = oh_start; oh <= oh_end; ++oh) {
                const int ih = oh * stride - padding + kh;
                const int64_t inRow = inB + (int64_t)ih * in_width;
                const int64_t goRow = goB + (int64_t)oh * out_width;

                for (int ow = ow_start; ow <= ow_end; ++ow) {
                    const int iw = ow * stride - padding + kw;
                    sum += ro(input + inRow + iw) * ro(grad_output + goRow + ow);
                }
            }
        }

        grad_weights[idx] = sum / static_cast<float>(batch_size);
    }
}

__global__ void conv2d_backward_bias(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_bias,
    int batch_size, int out_channels, int out_height, int out_width)
{
    // One thread per oc, grid-stride over N= B*H*W
    const int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= out_channels) return;

    const int64_t outHW = (int64_t)out_height * out_width;
    const int64_t perB  = (int64_t)out_channels * outHW;
    const int64_t N     = (int64_t)batch_size * outHW;

    float sum = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
        const int b  = n / outHW;
        const int64_t goB = (int64_t)b * perB + (int64_t)oc * outHW;
        const int64_t offs = n - (int64_t)b * outHW;
        sum += ro(grad_output + goB + offs);
    }
    grad_bias[oc] = sum / static_cast<float>(batch_size);
}

void launchConv2DForward(
    const float* input, const float* weights, const float* bias,
    float* output, int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
    cudaStream_t stream)
{
    const int64_t N = (int64_t)batch_size * out_channels * out_height * out_width;
    const int blockSize = 256;
    const int gridSize  = (int)std::min<int64_t>((N + blockSize - 1) / blockSize, 65535);
    conv2d_forward<<<gridSize, blockSize, 0, stream>>>(
        input, weights, bias, output,
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_size, stride, padding,
        out_height, out_width);
    CUDA_CHECK(cudaGetLastError());
}

void launchConv2DBackwardInput(
    const float* grad_output, const float* weights,
    float* grad_input, int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
    cudaStream_t stream)
{
    const int64_t N = (int64_t)batch_size * in_channels * in_height * in_width;
    const int blockSize = 256;
    const int gridSize  = (int)std::min<int64_t>((N + blockSize - 1) / blockSize, 65535);
    conv2d_backward_input<<<gridSize, blockSize, 0, stream>>>(
        grad_output, weights, grad_input,
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_size, stride, padding,
        out_height, out_width);
    CUDA_CHECK(cudaGetLastError());
}

void launchConv2DBackwardWeights(
    const float* input, const float* grad_output,
    float* grad_weights, int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
    cudaStream_t stream)
{
    const int64_t N = (int64_t)out_channels * in_channels * kernel_size * kernel_size;
    const int blockSize = 256;
    const int gridSize  = (int)std::min<int64_t>((N + blockSize - 1) / blockSize, 65535);
    conv2d_backward_weights<<<gridSize, blockSize, 0, stream>>>(
        input, grad_output, grad_weights,
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_size, stride, padding,
        out_height, out_width);
    CUDA_CHECK(cudaGetLastError());
}

void launchConv2DBackwardBias(
    const float* grad_output, float* grad_bias,
    int batch_size, int out_channels, int out_height, int out_width,
    cudaStream_t stream)
{
    const int blockSize = 256;
    const int gridSize  = (out_channels + blockSize - 1) / blockSize;
    conv2d_backward_bias<<<gridSize, blockSize, 0, stream>>>(
        grad_output, grad_bias,
        batch_size, out_channels, out_height, out_width);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace layers
} // namespace cuda