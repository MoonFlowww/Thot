#pragma once

#include <cuda_runtime.h>

namespace cuda::attentions {


    void launchMHAForward(
        const float* input,
        const float* Wq,
        const float* Wk,
        const float* Wv,
        const float* Wo,
        float* output,
        float* q_cache,
        float* k_cache,
        float* v_cache,
        float* attn_cache,
        float* context_cache,
        int batch_size,
        int seq_len,
        int embed_dim,
        int num_heads,
        cudaStream_t stream = 0);

    void launchMHABackward(
        const float* grad_output,
        const float* input,
        const float* Wq,
        const float* Wk,
        const float* Wv,
        const float* Wo,
        const float* q_cache,
        const float* k_cache,
        const float* v_cache,
        const float* attn_cache,
        const float* context_cache,
        float* grad_input,
        float* grad_Wq,
        float* grad_Wk,
        float* grad_Wv,
        float* grad_Wo,
        int batch_size,
        int seq_len,
        int embed_dim,
        int num_heads,
        cudaStream_t stream = 0);


}