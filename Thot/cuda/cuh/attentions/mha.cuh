#pragma once

#include <cuda_runtime.h>

namespace cuda::attentions {

    // Forward pass for Multi-Head Attention. Data resides on the GPU, but the
    // computation is carried out on the host for simplicity. The interface
    // mirrors other CUDA helpers in the project.
    void launchMHAForward(const float *input,
                          const float *w_q, const float *b_q,
                          const float *w_k, const float *b_k,
                          const float *w_v, const float *b_v,
                          const float *w_o, const float *b_o,
                          float *output,
                          float *q, float *k, float *v,
                          float *attn_probs, float *concat,
                          int batch_size, int seq_len, int embed_dim, int num_heads,
                          cudaStream_t stream = 0);

    // Backward pass for Multi-Head Attention producing gradients for inputs and
    // all projection matrices/biases.
    void launchMHABackward(const float *input,
                           const float *w_q, const float *b_q,
                           const float *w_k, const float *b_k,
                           const float *w_v, const float *b_v,
                           const float *w_o, const float *b_o,
                           const float *q, const float *k, const float *v,
                           const float *attn_probs, const float *concat,
                           const float *grad_output,
                           float *grad_input,
                           float *grad_w_q, float *grad_b_q,
                           float *grad_w_k, float *grad_b_k,
                           float *grad_w_v, float *grad_b_v,
                           float *grad_w_o, float *grad_b_o,
                           int batch_size, int seq_len, int embed_dim, int num_heads,
                           cudaStream_t stream = 0);

} // namespace cuda::attentions