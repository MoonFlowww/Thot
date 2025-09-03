#pragma once

#include <cuda_runtime.h>

namespace cuda::attentions {

    // Linear projection: input -> output using weight matrix
    __global__ void mha_linear(const float* input, const float* weights,
                               float* output, int total, int embed_dim);

    // Scaled dot-product attention producing context and softmax scores
    __global__ void mha_scaled_dot_product(const float* Q, const float* K, const float* V,
                                           float* softmax, float* context,
                                           int batch_size, int seq_len,
                                           int num_heads, int head_dim);

    // Simple kernel to zero gradients or buffers
    __global__ void mha_zero(float* data, int size);

    // Host launchers
    void launchMHAForward(const float* input,
                          const float* Wq, const float* Wk, const float* Wv, const float* Wo,
                          float* Q, float* K, float* V,
                          float* softmax, float* context, float* output,
                          int batch_size, int seq_len, int embed_dim, int num_heads,
                          cudaStream_t stream = 0);

    void launchMHABackward(const float* grad_output,
                           float* grad_input,
                           float* grad_Wq, float* grad_Wk,
                           float* grad_Wv, float* grad_Wo,
                           int total_tokens, int embed_dim,
                           cudaStream_t stream = 0);


} // namespace cuda