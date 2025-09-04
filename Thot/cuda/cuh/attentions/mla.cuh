#pragma once
#include <cuda_runtime.h>

namespace cuda::attentions {

void launchMLAForward(const float *input,
                      const float *W_DKV, const float *b_DKV,
                      const float *W_UK,  const float *b_UK,
                      const float *W_UV,  const float *b_UV,
                      const float *W_Q,   const float *b_Q,
                      const float *W_O,   const float *b_O,
                      float *output,
                      float *q, float *k, float *v,
                      float *c_kv,
                      float *attn_probs,
                      float *concat,
                      int batch_size, int seq_len, int embed_dim, int num_heads,
                      int latent_dim,
                      cudaStream_t stream = 0);


    void initMLABackwardWorkspace(int batch_size, int seq_len, int embed_dim,
                                  int num_heads, int latent_dim);

    void freeMLABackwardWorkspace();


void launchMLABackward(const float *input,
                       const float *W_DKV, const float *b_DKV,
                       const float *W_UK,  const float *b_UK,
                       const float *W_UV,  const float *b_UV,
                       const float *W_Q,   const float *b_Q,
                       const float *W_O,   const float *b_O,
                       const float *q, const float *k, const float *v,
                       const float *c_kv,
                       const float *attn_probs, const float *concat,
                       const float *grad_output,
                       float *grad_input,
                       float *grad_W_DKV, float *grad_b_DKV,
                       float *grad_W_UK,  float *grad_b_UK,
                       float *grad_W_UV,  float *grad_b_UV,
                       float *grad_W_Q,   float *grad_b_Q,
                       float *grad_W_O,   float *grad_b_O,
                       int batch_size, int seq_len, int embed_dim, int num_heads,
                       int latent_dim,
                       cudaStream_t stream = 0);

} // namespace cuda::attentions
