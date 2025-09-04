#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdio>

#include "../../cuh/attentions/mla.cuh"

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); } } while(0)
#endif

namespace cuda::attentions {

    __device__ inline int idx_bt(int b, int t, int e, int seq, int embed) {
        return (b * seq + t) * embed + e;
    }
    __device__ inline int idx_bthd(int b, int t, int h, int d, int seq, int embed, int head_dim) {
        return (b * seq + t) * embed + h * head_dim + d;
    }
    __device__ inline int idx_bhtt(int b, int h, int t, int t2, int seq, int num_heads) {
        return ((b * num_heads + h) * seq + t) * seq + t2;
    }




    ////////////////////////////////////////////////////////////////////////////////
    // Forward
    ////////////////////////////////////////////////////////////////////////////////


    __global__ void MLAForwardKernel(const float *input,
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
                                 int batch_size, int seq_len, int embed_dim,
                                 int num_heads, int latent_dim) {
    const int head_dim = embed_dim / num_heads;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    // loops over entire batch/sequence executed by single thread
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            // C = W_DKV * X + b_DKV
            for (int oc = 0; oc < latent_dim; ++oc) {
                float s = b_DKV ? b_DKV[oc] : 0.0f;
                for (int i = 0; i < embed_dim; ++i) {
                    s += W_DKV[oc * embed_dim + i] * input[idx_bt(b, t, i, seq_len, embed_dim)];
                    c_kv[(b * seq_len + t) * latent_dim + oc] = s;
                }
                // Q = W_Q * X + b_Q
                for (int o = 0; o < embed_dim; ++o) {
                    float s = b_Q ? b_Q[o] : 0.0f;
                    for (int i = 0; i < embed_dim; ++i) {
                        s += W_Q[o * embed_dim + i] * input[idx_bt(b, t, i, seq_len, embed_dim)];
                    }
                    q[idx_bt(b, t, o, seq_len, embed_dim)] = s;
                }
            }
        }

        // K,V from latent C
        for (int b = 0; b < batch_size; ++b) {
            for (int t = 0; t < seq_len; ++t) {
                for (int o = 0; o < embed_dim; ++o) {
                    float sk = b_UK ? b_UK[o] : 0.0f;
                    float sv = b_UV ? b_UV[o] : 0.0f;
                    for (int i = 0; i < latent_dim; ++i) {
                        float c = c_kv[(b * seq_len + t) * latent_dim + i];
                        sk += W_UK[o * latent_dim + i] * c;
                        sv += W_UV[o * latent_dim + i] * c;
                    }
                    k[idx_bt(b, t, o, seq_len, embed_dim)] = sk;
                    v[idx_bt(b, t, o, seq_len, embed_dim)] = sv;
                }
            }
        }
        // Attention and concat
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < num_heads; ++h) {
                for (int t = 0; t < seq_len; ++t) {
                    // compute raw scores in attn_probs
                    for (int t2 = 0; t2 < seq_len; ++t2) {
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; ++d) {
                            dot += q[idx_bthd(b, t, h, d, seq_len, embed_dim, head_dim)] *
                               k[idx_bthd(b, t2, h, d, seq_len, embed_dim, head_dim)];
                        }
                        attn_probs[idx_bhtt(b, h, t, t2, seq_len, num_heads)] =
                        dot * scale;
                    }
                    // softmax in-place
                    float mx = attn_probs[idx_bhtt(b, h, t, 0, seq_len, num_heads)];
                    for (int t2 = 1; t2 < seq_len; ++t2) {
                        float val = attn_probs[idx_bhtt(b, h, t, t2, seq_len, num_heads)];
                        mx = mx > val ? mx : val;
                    }
                    float sum = 0.0f;
                    for (int t2 = 0; t2 < seq_len; ++t2) {
                        int id = idx_bhtt(b, h, t, t2, seq_len, num_heads);
                        float e = expf(attn_probs[id] - mx);
                        attn_probs[id] = e;
                        sum += e;
                    }
                    for (int t2 = 0; t2 < seq_len; ++t2) {
                        int id = idx_bhtt(b, h, t, t2, seq_len, num_heads);
                        attn_probs[id] /= sum;
                    }
                    // concat = attn_probs * V
                    for (int d = 0; d < head_dim; ++d) {
                        float acc = 0.0f;
                        for (int t2 = 0; t2 < seq_len; ++t2) {
                            float p = attn_probs[idx_bhtt(b, h, t, t2, seq_len, num_heads)];
                            acc += p * v[idx_bthd(b, t2, h, d, seq_len, embed_dim, head_dim)];
                        }
                        concat[idx_bthd(b, t, h, d, seq_len, embed_dim, head_dim)] = acc;
                    }
                }
            }
        }

        // Output projection
        for (int b = 0; b < batch_size; ++b) {
            for (int t = 0; t < seq_len; ++t) {
                for (int o = 0; o < embed_dim; ++o) {
                    float s = b_O ? b_O[o] : 0.0f;
                    for (int i = 0; i < embed_dim; ++i) {
                        s += W_O[o * embed_dim + i] *
                         concat[idx_bt(b, t, i, seq_len, embed_dim)];
                    }
                    output[idx_bt(b, t, o, seq_len, embed_dim)] = s;
                }
            }
        }
    }

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
                      cudaStream_t stream) {
        MLAForwardKernel<<<1, 1, 0, stream>>>(input, W_DKV, b_DKV,
                                              W_UK, b_UK,
                                              W_UV, b_UV,
                                              W_Q, b_Q,
                                              W_O, b_O,
                                              output, q, k, v,
                                              c_kv, attn_probs, concat,
                                              batch_size, seq_len, embed_dim,
                                              num_heads, latent_dim);
        CUDA_CHECK(cudaGetLastError());
    }
    ////////////////////////////////////////////////////////////////////////////////
    // Backward
    ////////////////////////////////////////////////////////////////////////////////

    __global__ void MLABackwardKernel(const float *input,
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
                                      float *dConcat, float *dAttn,
                                      float *dV, float *dK, float *dQ, float *dC,
                                      int batch_size, int seq_len, int embed_dim,
                                      int num_heads, int latent_dim) {
        const int head_dim = embed_dim / num_heads;

        size_t X_size = static_cast<size_t>(batch_size) * seq_len * embed_dim;
        size_t C_size = static_cast<size_t>(batch_size) * seq_len * latent_dim;
        size_t prob_sz = static_cast<size_t>(batch_size) * num_heads * seq_len * seq_len;
        // zero gradients and temporaries
        for (size_t i = 0; i < X_size; ++i) {
            grad_input[i] = 0.0f;
            dConcat[i] = 0.0f;
            dV[i] = 0.0f;
            dK[i] = 0.0f;
            dQ[i] = 0.0f;
        }
        for (size_t i = 0; i < C_size; ++i) dC[i] = 0.0f;
        for (size_t i = 0; i < prob_sz; ++i) dAttn[i] = 0.0f;
        size_t wo = static_cast<size_t>(embed_dim) * embed_dim;
        size_t wuk = static_cast<size_t>(embed_dim) * latent_dim;
        for (size_t i = 0; i < wo; ++i) {
            grad_W_O[i] = 0.0f;
            grad_W_Q[i] = 0.0f;
        }
        for (size_t i = 0; i < wuk; ++i) {
            grad_W_DKV[i] = 0.0f;
            grad_W_UK[i] = 0.0f;
            grad_W_UV[i] = 0.0f;
        }
        for (int i = 0; i < embed_dim; ++i) {
            grad_b_O[i] = 0.0f;
            grad_b_Q[i] = 0.0f;
            grad_b_UK[i] = 0.0f;
            grad_b_UV[i] = 0.0f;
        }
        for (int i = 0; i < latent_dim; ++i) {
            grad_b_DKV[i] = 0.0f;
        }

        // dOut -> gW_O, gB_O, dConcat
        for (int b = 0; b < batch_size; ++b) {
            for (int t = 0; t < seq_len; ++t) {
                for (int o = 0; o < embed_dim; ++o) {
                    float go = grad_output[idx_bt(b, t, o, seq_len, embed_dim)];
                    grad_b_O[o] += go;
                    for (int i = 0; i < embed_dim; ++i) {
                        grad_W_O[o * embed_dim + i] +=
                        go * concat[idx_bt(b, t, i, seq_len, embed_dim)];
                        dConcat[idx_bt(b, t, i, seq_len, embed_dim)] +=
                            go * W_O[o * embed_dim + i];
                    }
                }
            }
        }

        // back through attention: dConcat -> dV, dAttn
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < num_heads; ++h) {
                for (int t = 0; t < seq_len; ++t) {
                    for (int d = 0; d < head_dim; ++d) {
                        float gc = dConcat[idx_bthd(b, t, h, d, seq_len, embed_dim, head_dim)];
                        for (int t2 = 0; t2 < seq_len; ++t2) {
                            float p = attn_probs[idx_bhtt(b, h, t, t2, seq_len, num_heads)];
                            dV[idx_bthd(b, t2, h, d, seq_len, embed_dim, head_dim)] += p * gc;
                            dAttn[idx_bhtt(b, h, t, t2, seq_len, num_heads)] +=
                                gc * v[idx_bthd(b, t2, h, d, seq_len, embed_dim, head_dim)];
                        }
                    }
                }
            }
        }

        // softmax backward -> dQ, dK
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < num_heads; ++h) {
                for (int t = 0; t < seq_len; ++t) {
                    float sum = 0.0f;
                    for (int t2 = 0; t2 < seq_len; ++t2) {
                        int id = idx_bhtt(b, h, t, t2, seq_len, num_heads);
                        sum += dAttn[id] * attn_probs[id];
                    }
                    for (int t2 = 0; t2 < seq_len; ++t2) {
                        int id = idx_bhtt(b, h, t, t2, seq_len, num_heads);
                        float grad = attn_h[id] * (dAttn_h[id] - sum);
                        grad *= 1.0f / std::sqrt(static_cast<float>(head_dim));
                        float grad = attn_probs[id] * (dAttn[id] - sum);
                        grad *= 1.0f / sqrtf(static_cast<float>(head_dim));
                        for (int d = 0; d < head_dim; ++d) {
                            dQ[idx_bthd(b, t, h, d, seq_len, embed_dim, head_dim)] +=
                            grad * k[idx_bthd(b, t2, h, d, seq_len, embed_dim, head_dim)];
                            dK[idx_bthd(b, t2, h, d, seq_len, embed_dim, head_dim)] +=
                                grad * q[idx_bthd(b, t, h, d, seq_len, embed_dim, head_dim)];
                        }
                    }
                }
            }
        }
        // dQ -> gW_Q,gB_Q,dX
        for (int b = 0; b < batch_size; ++b) {
            for (int t = 0; t < seq_len; ++t) {
                for (int o = 0; o < embed_dim; ++o) {
                    float gq = dQ[idx_bt(b, t, o, seq_len, embed_dim)];
                    grad_b_Q[o] += gq;
                    for (int i = 0; i < embed_dim; ++i) {
                        float x = input[idx_bt(b, t, i, seq_len, embed_dim)];
                        grad_W_Q[o * embed_dim + i] += gq * x;
                        grad_input[idx_bt(b, t, i, seq_len, embed_dim)] +=
                            gq * W_Q[o * embed_dim + i];
                    }
                }
            }
        }

        // dK,dV -> dC
        for (int b = 0; b < batch_size; ++b) {
            for (int t = 0; t < seq_len; ++t) {
                for (int i = 0; i < latent_dim; ++i) {
                    float s = 0.0f;
                    for (int o = 0; o < embed_dim; ++o) {
                        s += W_UK[o * latent_dim + i] * dK[idx_bt(b, t, o, seq_len, embed_dim)];
                        s += W_UV[o * latent_dim + i] * dV[idx_bt(b, t, o, seq_len, embed_dim)];
                    }
                    dC[(b * seq_len + t) * latent_dim + i] = s;
                }
            }
        }
        // accumulate gradients for UK/UV and biases
        for (int b = 0; b < batch_size; ++b) {
            for (int t = 0; t < seq_len; ++t) {
                for (int o = 0; o < embed_dim; ++o) {
                    float dk = dK[idx_bt(b, t, o, seq_len, embed_dim)];
                    float dv = dV[idx_bt(b, t, o, seq_len, embed_dim)];
                    grad_b_UK[o] += dk;
                    grad_b_UV[o] += dv;
                    for (int i = 0; i < latent_dim; ++i) {
                        float c = c_kv[(b * seq_len + t) * latent_dim + i];
                        grad_W_UK[o * latent_dim + i] += dk * c;
                        grad_W_UV[o * latent_dim + i] += dv * c;
                    }
                }
            }
        }

        // dC -> dX
        for (int b = 0; b < batch_size; ++b) {
            for (int t = 0; t < seq_len; ++t) {
                for (int i = 0; i < embed_dim; ++i) {
                    float s = 0.0f;
                    for (int oc = 0; oc < latent_dim; ++oc) {
                        s += W_DKV[oc * embed_dim + i] *
                         dC[(b * seq_len + t) * latent_dim + oc];
                    }
                    grad_input[idx_bt(b, t, i, seq_len, embed_dim)] += s;
                }
            }
        }
        // accumulate gradients for W_DKV/b_DKV
        for (int b = 0; b < batch_size; ++b) {
            for (int t = 0; t < seq_len; ++t) {
                for (int oc = 0; oc < latent_dim; ++oc) {
                    float dc = dC[(b * seq_len + t) * latent_dim + oc];
                    grad_b_DKV[oc] += dc;
                    for (int i = 0; i < embed_dim; ++i) {
                        grad_W_DKV[oc * embed_dim + i] +=
                        dc * input[idx_bt(b, t, i, seq_len, embed_dim)];
                    }
                }
            }
        }
    }
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
                       cudaStream_t stream) {
        size_t X_size = static_cast<size_t>(batch_size) * seq_len * embed_dim;
        size_t C_size = static_cast<size_t>(batch_size) * seq_len * latent_dim;
        size_t prob_sz = static_cast<size_t>(batch_size) * num_heads * seq_len * seq_len;

        float *dConcat = nullptr, *dAttn = nullptr, *dV = nullptr, *dK = nullptr, *dQ = nullptr, *dC = nullptr;
        CUDA_CHECK(cudaMalloc(&dConcat, X_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dAttn,  prob_sz * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dV,     X_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dK,     X_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dQ,     X_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC,     C_size * sizeof(float)));

        MLABackwardKernel<<<1, 1, 0, stream>>>(input, W_DKV, b_DKV,
                                               W_UK, b_UK,
                                               W_UV, b_UV,
                                               W_Q, b_Q,
                                               W_O, b_O,
                                               q, k, v, c_kv,
                                               attn_probs, concat,
                                               grad_output,
                                               grad_input,
                                               grad_W_DKV, grad_b_DKV,
                                               grad_W_UK, grad_b_UK,
                                               grad_W_UV, grad_b_UV,
                                               grad_W_Q, grad_b_Q,
                                               grad_W_O, grad_b_O,
                                               dConcat, dAttn, dV, dK, dQ, dC,
                                               batch_size, seq_len, embed_dim,
                                               num_heads, latent_dim);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(dConcat));
        CUDA_CHECK(cudaFree(dAttn));
        CUDA_CHECK(cudaFree(dV));
        CUDA_CHECK(cudaFree(dK));
        CUDA_CHECK(cudaFree(dQ));
        CUDA_CHECK(cudaFree(dC));
    }
} // namespace cuda::attention
