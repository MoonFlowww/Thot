#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>
#include "../../cuh/attentions/mha.cuh"

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

namespace cuda::attention {


    __global__ void mha_linear(const float* __restrict__ input,
                               const float* __restrict__ weights,
                               float* __restrict__ output,
                               int total, int embed_dim) {
        const int64_t N = (int64_t)total * embed_dim;
        for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < N;
             idx += (int64_t)blockDim.x * gridDim.x) {
            int64_t t = idx;
            const int row = t / embed_dim;        // token index
            const int col = t - (int64_t)row * embed_dim; // output dim
            const float* in_row = input + (int64_t)row * embed_dim;
            const float* w_col = weights + col;
            float sum = 0.0f;
            for (int k = 0; k < embed_dim; ++k) {
                sum += ro(in_row + k) * ro(w_col + (int64_t)k * embed_dim);
            }
            output[idx] = sum;
        }
    }

    __global__ void mha_scaled_dot_product(const float* __restrict__ Q,
                                           const float* __restrict__ K,
                                           const float* __restrict__ V,
                                           float* __restrict__ softmax,
                                           float* __restrict__ context,
                                           int batch_size, int seq_len,
                                           int num_heads, int head_dim) {
        const int b = blockIdx.x;
        const int h = blockIdx.y;
        if (b >= batch_size || h >= num_heads) return;
        const int embed_dim = num_heads * head_dim;

        const float scale = 1.f / sqrtf((float)head_dim);

        const float* q = Q + ((b * num_heads + h) * seq_len * head_dim);
        const float* k = K + ((b * num_heads + h) * seq_len * head_dim);
        const float* v = V + ((b * num_heads + h) * seq_len * head_dim);
        float* sm = softmax + ((b * num_heads + h) * seq_len * seq_len);

        // context for this head will be written into global concatenated buffer
        float* out = context + (b * seq_len * embed_dim) + h * head_dim;

        for (int i = 0; i < seq_len; ++i) {
            // compute scores and store in sm temporarily
            float max_val = -1e20f;
            for (int j = 0; j < seq_len; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += ro(q + i * head_dim + d) * ro(k + j * head_dim + d);
                }
                dot *= scale;
                sm[i * seq_len + j] = dot;
                if (dot > max_val) max_val = dot;
            }
            // softmax
            float denom = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                float e = expf(sm[i * seq_len + j] - max_val);
                sm[i * seq_len + j] = e;
                denom += e;
            }
            for (int j = 0; j < seq_len; ++j) {
                sm[i * seq_len + j] /= denom;
            }
            // context vector for position i
            for (int d = 0; d < head_dim; ++d) {
                float val = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    val += sm[i * seq_len + j] * ro(v + j * head_dim + d);
                }
                out[i * embed_dim + d] = val; // position i, head h, dim d
            }
        }
    }

    __global__ void mha_zero(float* data, int size) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) data[idx] = 0.0f;
    }

    void launchMHAForward(const float* input,
                          const float* Wq, const float* Wk, const float* Wv, const float* Wo,
                          float* Q, float* K, float* V,
                          float* softmax, float* context, float* output,
                          int batch_size, int seq_len, int embed_dim, int num_heads,
                          cudaStream_t stream) {
        const int total_tokens = batch_size * seq_len;
        const int blockSize = 256;
        int64_t N = (int64_t)total_tokens * embed_dim;
        const int gridSize = (int)std::min<int64_t>((N + blockSize - 1) / blockSize, 65535);

        mha_linear<<<gridSize, blockSize, 0, stream>>>(input, Wq, Q, total_tokens, embed_dim);
        mha_linear<<<gridSize, blockSize, 0, stream>>>(input, Wk, K, total_tokens, embed_dim);
        mha_linear<<<gridSize, blockSize, 0, stream>>>(input, Wv, V, total_tokens, embed_dim);
        CUDA_CHECK(cudaGetLastError());

        dim3 grid_sd(batch_size, num_heads);
        mha_scaled_dot_product<<<grid_sd, 1, 0, stream>>>(Q, K, V, softmax, context,
                                                          batch_size, seq_len, num_heads,
                                                          embed_dim / num_heads);
        CUDA_CHECK(cudaGetLastError());

        mha_linear<<<gridSize, blockSize, 0, stream>>>(context, Wo, output, total_tokens, embed_dim);
        CUDA_CHECK(cudaGetLastError());
    }

    void launchMHABackward(const float* grad_output,
                           float* grad_input,
                           float* grad_Wq, float* grad_Wk,
                           float* grad_Wv, float* grad_Wo,
                           int total_tokens, int embed_dim,
                           cudaStream_t stream) {
        int data_size = total_tokens * embed_dim;
        const int blockSize = 256;
        int grid = (data_size + blockSize - 1) / blockSize;
        mha_zero<<<grid, blockSize, 0, stream>>>(grad_input, data_size);
        int w_size = embed_dim * embed_dim;
        int gridW = (w_size + blockSize - 1) / blockSize;
        mha_zero<<<gridW, blockSize, 0, stream>>>(grad_Wq, w_size);
        mha_zero<<<gridW, blockSize, 0, stream>>>(grad_Wk, w_size);
        mha_zero<<<gridW, blockSize, 0, stream>>>(grad_Wv, w_size);
        mha_zero<<<gridW, blockSize, 0, stream>>>(grad_Wo, w_size);
        CUDA_CHECK(cudaGetLastError());
    }


} // namespace cuda