#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include "../../cuh/layers/rnn.cuh"

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
    __global__ void rnn_forward(
        const float* __restrict__ input, const float* __restrict__ weights_ih, const float* __restrict__ weights_hh, const float* __restrict__ bias,
        const float* __restrict__ prev_hidden_state, float* __restrict__ hidden_state, float* __restrict__ output, int batch_size, int seq_length, int input_size, int hidden_size) {
        const int64_t N = (int64_t)batch_size * hidden_size;
        for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < N;
             idx += (int64_t)blockDim.x * gridDim.x)
        {
            int64_t t = idx;
            const int b = t / hidden_size;
            const int h = t - (int64_t)b * hidden_size;

            float sum = 0.0f;
            const int64_t inB = (int64_t)b * input_size;
            const int64_t wihRow = (int64_t)h * input_size;
            for (int i = 0; i < input_size; ++i) {
                sum += ro(input + inB + i) * ro(weights_ih + wihRow + i);
            }
            const int64_t prevB = (int64_t)b * hidden_size;
            const int64_t whhRow = (int64_t)h * hidden_size;
            for (int hh = 0; hh < hidden_size; ++hh) {
                sum += ro(prev_hidden_state + prevB + hh) * ro(weights_hh + whhRow + hh);
            }
            if (bias) sum += ro(bias + h);

            float activated = tanhf(sum);
            hidden_state[idx] = activated;
            output[idx] = activated;
        }
    }

    __global__ void rnn_backward_input(
    const float* __restrict__ grad_output, const float* __restrict__ weights_ih, float* __restrict__ grad_input, int batch_size, int seq_length, int input_size, int hidden_size) {
        const int64_t N = (int64_t)batch_size * input_size;
        for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < N;
             idx += (int64_t)blockDim.x * gridDim.x)
        {
            int64_t t = idx;
            const int b = t / input_size;
            const int i = t - (int64_t)b * input_size;

            float sum = 0.0f;
            const int64_t goB = (int64_t)b * hidden_size;
            for (int h = 0; h < hidden_size; ++h) {
                sum += ro(grad_output + goB + h) * ro(weights_ih + (int64_t)h * input_size + i);
            }
            grad_input[idx] = sum / static_cast<float>(batch_size);
        }
    }
    __global__ void rnn_backward_hidden(const float* __restrict__ grad_output, const float* __restrict__ weights_hh, float* __restrict__ grad_hidden, int batch_size, int hidden_size) {
        const int64_t N = (int64_t)batch_size * hidden_size;
        for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < N;
             idx += (int64_t)blockDim.x * gridDim.x)
        {
            int64_t t = idx;
            const int b = t / hidden_size;
            const int h = t - (int64_t)b * hidden_size;

            float sum = 0.0f;
            const int64_t goB = (int64_t)b * hidden_size;
            for (int hh = 0; hh < hidden_size; ++hh) {
                sum += ro(grad_output + goB + hh) * ro(weights_hh + (int64_t)hh * hidden_size + h);
            }
            grad_hidden[idx] = sum / static_cast<float>(batch_size);
        }
    }
    __global__ void rnn_backward_weights_ih(const float* __restrict__ input, const float* __restrict__ grad_hidden, float* __restrict__ grad_weights_ih, int batch_size, int seq_length, int input_size, int hidden_size) {
        const int64_t N = (int64_t)hidden_size * input_size;
        for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < N;
             idx += (int64_t)blockDim.x * gridDim.x)
        {
            int64_t t = idx;
            const int h = t / input_size;
            const int i = t - (int64_t)h * input_size;

            float sum = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                const int64_t inIdx = (int64_t)b * input_size + i;
                const int64_t ghIdx = (int64_t)b * hidden_size + h;
                sum += ro(input + inIdx) * ro(grad_hidden + ghIdx);
            }
            grad_weights_ih[idx] += sum;
        }
    }

    __global__ void rnn_backward_weights_hh( const float* __restrict__ hidden_state, const float* __restrict__ grad_hidden, float* __restrict__ grad_weights_hh, int batch_size, int hidden_size) {
        const int64_t N = (int64_t)hidden_size * hidden_size;
        for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < N;
             idx += (int64_t)blockDim.x * gridDim.x)
        {
            int64_t t = idx;
            const int h1 = t / hidden_size;
            const int h2 = t - (int64_t)h1 * hidden_size;

            float sum = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                const int64_t hsIdx = (int64_t)b * hidden_size + h2;
                const int64_t ghIdx = (int64_t)b * hidden_size + h1;
                sum += ro(hidden_state + hsIdx) * ro(grad_hidden + ghIdx);
            }
            grad_weights_hh[idx] += sum;
        }
    }
    __global__ void rnn_backward_bias(const float* __restrict__ grad_hidden, float* __restrict__ grad_bias, int batch_size, int hidden_size) {
        const int oc = blockIdx.x * blockDim.x + threadIdx.x;
        if (oc >= hidden_size) return;

        float sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            sum += ro(grad_hidden + (int64_t)b * hidden_size + oc);
        }
        grad_bias[oc] += sum;
    }

    void launchRNNForward(
    const float* input, const float* weights_ih, const float* weights_hh, const float* bias, const float* prev_hidden_state, float* hidden_state, float* output, int batch_size, int seq_length, int input_size, int hidden_size, cudaStream_t stream) {
        const int64_t N = (int64_t)batch_size * hidden_size;
        const int blockSize = 256;
        const int gridSize = (int)std::min<int64_t>((N + blockSize - 1) / blockSize, 65535);
        rnn_forward<<<gridSize, blockSize, 0, stream>>>(
            input, weights_ih, weights_hh, bias,
            prev_hidden_state, hidden_state, output,
            batch_size, seq_length, input_size, hidden_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void launchRNNBackwardInput(
    const float* grad_output, const float* weights_ih, float* grad_input, int batch_size, int seq_length, int input_size, int hidden_size, cudaStream_t stream) {
        const int64_t N = (int64_t)batch_size * input_size;
        const int blockSize = 256;
        const int gridSize = (int)std::min<int64_t>((N + blockSize - 1) / blockSize, 65535);
        rnn_backward_input<<<gridSize, blockSize, 0, stream>>>(
            grad_output, weights_ih, grad_input,
            batch_size, seq_length, input_size, hidden_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void launchRNNBackwardHidden(const float* grad_output, const float* weights_hh, float* grad_hidden, int batch_size, int hidden_size, cudaStream_t stream) {
        const int64_t N = (int64_t)batch_size * hidden_size;
        const int blockSize = 256;
        const int gridSize = (int)std::min<int64_t>((N + blockSize - 1) / blockSize, 65535);
        rnn_backward_hidden<<<gridSize, blockSize, 0, stream>>>(
            grad_output, weights_hh, grad_hidden,
            batch_size, hidden_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void launchRNNBackwardWeightsIH(const float* input, const float* grad_hidden, float* grad_weights_ih, int batch_size, int seq_length, int input_size, int hidden_size, cudaStream_t stream){
        const int64_t N = (int64_t)hidden_size * input_size;
        const int blockSize = 256;
        const int gridSize = (int)std::min<int64_t>((N + blockSize - 1) / blockSize, 65535);
        rnn_backward_weights_ih<<<gridSize, blockSize, 0, stream>>>(
            input, grad_hidden, grad_weights_ih,
            batch_size, seq_length, input_size, hidden_size);
        CUDA_CHECK(cudaGetLastError());
    }


    void launchRNNBackwardWeightsHH(const float* hidden_state, const float* grad_hidden, float* grad_weights_hh, int batch_size, int hidden_size, cudaStream_t stream){
        const int64_t N = (int64_t)hidden_size * hidden_size;
        const int blockSize = 256;
        const int gridSize = (int)std::min<int64_t>((N + blockSize - 1) / blockSize, 65535);
        rnn_backward_weights_hh<<<gridSize, blockSize, 0, stream>>>(
            hidden_state, grad_hidden, grad_weights_hh,
            batch_size, hidden_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void launchRNNBackwardBias(const float* grad_hidden, float* grad_bias, int batch_size, int hidden_size, cudaStream_t stream){
        const int blockSize = 256;
        const int gridSize = (hidden_size + blockSize - 1) / blockSize;
        rnn_backward_bias<<<gridSize, blockSize, 0, stream>>>(
            grad_hidden, grad_bias,
            batch_size, hidden_size);
        CUDA_CHECK(cudaGetLastError());
    }
}