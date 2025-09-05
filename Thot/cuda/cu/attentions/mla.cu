#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#include "../../cuh/attentions/mla.cuh"

#include <cfloat>

#ifndef CUDA_CHECK
#define CUDA_CHECK(x)                                                          \
    do {                                                                         \
        cudaError_t _e = (x);                                                      \
        if (_e != cudaSuccess) {                                                   \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,                     \
            cudaGetErrorString(_e));                                          \
        }                                                                          \
    } while (0)
#endif

namespace cuda::attentions {
    namespace {
        float *g_dConcat = nullptr;
        float *g_dAttn = nullptr;
        float *g_dV = nullptr;
        float *g_dK = nullptr;
        float *g_dQ = nullptr;
        float *g_dC = nullptr;
    }
    size_t g_X_size = 0;
    size_t g_C_size = 0;
    size_t g_prob_sz = 0;


    __host__ __device__ inline int idx_bt(int b, int t, int e, int seq, int embed) {
        return (b * seq + t) * embed + e;
    }
    __host__ __device__ inline int idx_bthd(int b, int t, int h, int d, int seq, int embed,
                                   int head_dim) {
        return (b * seq + t) * embed + h * head_dim + d;
    }
    __host__ __device__ inline int idx_bhtt(int b, int h, int t, int t2, int seq,
                                   int num_heads) {
        return ((b * num_heads + h) * seq + t) * seq + t2;
    }

    __device__ float blockReduceMax(float val, float* sdata) {
        int tid = threadIdx.x;
        sdata[tid] = val;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            __syncthreads();
        }
        return sdata[0];
    }

    __device__ float blockReduceSum(float val, float* sdata) {
        int tid = threadIdx.x;
        sdata[tid] = val;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        return sdata[0];
    }

    __global__ void MHAForwardKernel(const float* q, const float* k, const float* v,
                                     float* attn_probs, float* concat,
                                     int batch_size, int seq_len, int embed_dim, int num_heads) {
        int head_dim = embed_dim / num_heads;
        int bh = blockIdx.x;
        int b = bh / num_heads;
        int h = bh % num_heads;

        extern __shared__ float sh[];
        float* q_sh = sh;                   // head_dim
        float* scores = q_sh + head_dim;    // seq_len
        float* red = scores + seq_len;      // blockDim.x

        const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

        for (int t = 0; t < seq_len; ++t) {
            // load query vector
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
                q_sh[d] = q[idx_bthd(b, t, h, d, seq_len, embed_dim, head_dim)];
            __syncthreads();

            // compute raw scores
            for (int t2 = threadIdx.x; t2 < seq_len; t2 += blockDim.x) {
                const float* k_ptr = &k[idx_bthd(b, t2, h, 0, seq_len, embed_dim, head_dim)];
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += q_sh[d] * k_ptr[d];
                }
                scores[t2] = dot * scale;
            }
            __syncthreads();

            // compute max via block reduction
            float thread_max = -FLT_MAX;
            for (int t2 = threadIdx.x; t2 < seq_len; t2 += blockDim.x)
                thread_max = fmaxf(thread_max, scores[t2]);
            float max_val = blockReduceMax(thread_max, red);
            __syncthreads();

            // compute exp and sum
            float thread_sum = 0.0f;
            for (int t2 = threadIdx.x; t2 < seq_len; t2 += blockDim.x) {
                float e = expf(scores[t2] - max_val);
                scores[t2] = e;
                thread_sum += e;
            }
            float sum_val = blockReduceSum(thread_sum, red);
            __syncthreads();
            float inv_sum = 1.0f / sum_val;

            // accumulate weighted values into q_sh reused as buffer
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
                q_sh[d] = 0.0f;
            __syncthreads();
            for (int t2 = threadIdx.x; t2 < seq_len; t2 += blockDim.x) {
                float p = scores[t2] * inv_sum;
                attn_probs[idx_bhtt(b, h, t, t2, seq_len, num_heads)] = p;
                const float* v_ptr = &v[idx_bthd(b, t2, h, 0, seq_len, embed_dim, head_dim)];
                for (int d = 0; d < head_dim; ++d) {
                    atomicAdd(&q_sh[d], p * v_ptr[d]);
                }
            }
            __syncthreads();
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
                concat[idx_bthd(b, t, h, d, seq_len, embed_dim, head_dim)] = q_sh[d];
            __syncthreads();
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Forward
    ////////////////////////////////////////////////////////////////////////////////

    __global__ void
    MLAForwardKernel(const float *input, const float *W_DKV, const float *b_DKV,
                     const float *W_UK, const float *b_UK, const float *W_UV,
                     const float *b_UV, const float *W_Q, const float *b_Q,
                     const float *W_O, const float *b_O, float *output, float *q,
                     float *k, float *v, float *c_kv, float *attn_probs,
                     float *concat, int batch_size, int seq_len, int embed_dim,
                     int num_heads, int latent_dim) {
        const int head_dim = embed_dim / num_heads;
        const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
        const int total = batch_size * seq_len;
        const int stride = blockDim.x * gridDim.x;

        extern __shared__ float shmem[];

        // Phase 1: compute C and Q for each (b,t)
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
             idx += stride) {
            int b = idx / seq_len;
            int t = idx % seq_len;

            float *x = shmem + threadIdx.x * embed_dim;
            for (int e = 0; e < embed_dim; ++e) {
                x[e] = input[idx_bt(b, t, e, seq_len, embed_dim)];
            }
            for (int oc = 0; oc < latent_dim; ++oc) {
                float s = b_DKV ? b_DKV[oc] : 0.0f;
                for (int i = 0; i < embed_dim; ++i) {
                    s += W_DKV[oc * embed_dim + i] * x[i];
                }
                c_kv[(b * seq_len + t) * latent_dim + oc] = s;
            }

            for (int o = 0; o < embed_dim; ++o) {
                float s = b_Q ? b_Q[o] : 0.0f;
                for (int i = 0; i < embed_dim; ++i) {
                    s += W_Q[o * embed_dim + i] * x[i];
                }
                q[idx_bt(b, t, o, seq_len, embed_dim)] = s;
            }
             }

        __syncthreads();

        // Phase 2: compute K and V from latent C
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
             idx += stride) {
            int b = idx / seq_len;
            int t = idx % seq_len;
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
        __syncthreads();

        // Phase 3: attention and concat
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
             idx += stride) {
            int b = idx / seq_len;
            int t = idx % seq_len;
            for (int h = 0; h < num_heads; ++h) {
                for (int t2 = 0; t2 < seq_len; ++t2) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += q[idx_bthd(b, t, h, d, seq_len, embed_dim, head_dim)] *
                               k[idx_bthd(b, t2, h, d, seq_len, embed_dim, head_dim)];
                    }
                    attn_probs[idx_bhtt(b, h, t, t2, seq_len, num_heads)] = dot * scale;
                }

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
        __syncthreads();

        // Phase 4: output projection
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
             idx += stride) {
            int b = idx / seq_len;
            int t = idx % seq_len;

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

    void launchMLAForward(const float *input, const float *W_DKV,
                      const float *b_DKV, const float *W_UK, const float *b_UK,
                      const float *W_UV, const float *b_UV, const float *W_Q,
                      const float *b_Q, const float *W_O, const float *b_O,
                      float *output, float *q, float *k, float *v, float *c_kv,
                      float *attn_probs, float *concat, int batch_size,
                      int seq_len, int embed_dim, int num_heads, int latent_dim,
                      cudaStream_t stream) {
        int total = batch_size * seq_len;
        int threads = 256;
        int max_threads = 49152 / (embed_dim * static_cast<int>(sizeof(float)));
        if (threads > max_threads)
            threads = max_threads > 0 ? max_threads : 1;
        int blocks = (total + threads - 1) / threads;
        size_t shmem = static_cast<size_t>(threads) * embed_dim * sizeof(float);
        MLAForwardKernel<<<blocks, threads, shmem, stream>>>(
            input, W_DKV, b_DKV, W_UK, b_UK, W_UV, b_UV, W_Q, b_Q, W_O, b_O, output,
            q, k, v, c_kv, attn_probs, concat, batch_size, seq_len, embed_dim,
            num_heads, latent_dim);
        CUDA_CHECK(cudaGetLastError());
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Backward
    ////////////////////////////////////////////////////////////////////////////////

    void initMLABackwardWorkspace(int batch_size, int seq_len, int embed_dim,
                                  int num_heads, int latent_dim) {
        size_t X_size = static_cast<size_t>(batch_size) * seq_len * embed_dim;
        size_t C_size = static_cast<size_t>(batch_size) * seq_len * latent_dim;
        size_t prob_sz = static_cast<size_t>(batch_size) * num_heads * seq_len * seq_len;

        if (X_size > g_X_size) {
            if (g_dConcat)
                CUDA_CHECK(cudaFree(g_dConcat));
            if (g_dV)
                CUDA_CHECK(cudaFree(g_dV));
            if (g_dK)
                CUDA_CHECK(cudaFree(g_dK));
            if (g_dQ)
                CUDA_CHECK(cudaFree(g_dQ));
            CUDA_CHECK(cudaMalloc(&g_dConcat, X_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&g_dV, X_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&g_dK, X_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&g_dQ, X_size * sizeof(float)));
            g_X_size = X_size;
        }

        if (prob_sz > g_prob_sz) {
            if (g_dAttn)
                CUDA_CHECK(cudaFree(g_dAttn));
            CUDA_CHECK(cudaMalloc(&g_dAttn, prob_sz * sizeof(float)));
            g_prob_sz = prob_sz;
        }

        if (C_size > g_C_size) {
            if (g_dC)
                CUDA_CHECK(cudaFree(g_dC));
            CUDA_CHECK(cudaMalloc(&g_dC, C_size * sizeof(float)));
            g_C_size = C_size;
        }
    }

    void freeMLABackwardWorkspace() {
        if (g_dConcat) {
            CUDA_CHECK(cudaFree(g_dConcat));
            g_dConcat = nullptr;
        }
        if (g_dAttn) {
            CUDA_CHECK(cudaFree(g_dAttn));
            g_dAttn = nullptr;
        }
        if (g_dV) {
            CUDA_CHECK(cudaFree(g_dV));
            g_dV = nullptr;
        }
        if (g_dK) {
            CUDA_CHECK(cudaFree(g_dK));
            g_dK = nullptr;
        }
        if (g_dQ) {
            CUDA_CHECK(cudaFree(g_dQ));
            g_dQ = nullptr;
        }
        if (g_dC) {
            CUDA_CHECK(cudaFree(g_dC));
            g_dC = nullptr;
        }
        g_X_size = g_C_size = g_prob_sz = 0;
    }

    __global__ void MLABackwardKernel(
        const float *input, const float *W_DKV, const float *b_DKV,
        const float *W_UK, const float *b_UK, const float *W_UV, const float *b_UV,
        const float *W_Q, const float *b_Q, const float *W_O, const float *b_O,
        const float *q, const float *k, const float *v, const float *c_kv,
        const float *attn_probs, const float *concat, const float *grad_output,
        float *grad_input, float *grad_W_DKV, float *grad_b_DKV, float *grad_W_UK,
        float *grad_b_UK, float *grad_W_UV, float *grad_b_UV, float *grad_W_Q,
        float *grad_b_Q, float *grad_W_O, float *grad_b_O, float *dConcat,
        float *dAttn, float *dV, float *dK, float *dQ, float *dC, int batch_size,
        int seq_len, int embed_dim, int num_heads, int latent_dim) {

        const int head_dim = embed_dim / num_heads;
        const int total_tokens = batch_size * seq_len;
        const int stride = blockDim.x * gridDim.x;

        // zero per-token buffers
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < batch_size * seq_len * embed_dim; idx += stride) {
            grad_input[idx] = 0.0f;
            dConcat[idx]    = 0.0f;
            dV[idx]         = 0.0f;
            dK[idx]         = 0.0f;
            dQ[idx]         = 0.0f;

        }
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < batch_size * seq_len * latent_dim; idx += stride)
            dC[idx] = 0.0f;

        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < batch_size * num_heads * seq_len * seq_len; idx += stride)
            dAttn[idx] = 0.0f;

        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < embed_dim * embed_dim; idx += stride) {
            grad_W_O[idx] = 0.0f;
            grad_W_Q[idx] = 0.0f;
        }

        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < embed_dim * latent_dim; idx += stride) {
            grad_W_DKV[idx] = 0.0f;
            grad_W_UK[idx]  = 0.0f;
            grad_W_UV[idx]  = 0.0f;
        }

        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < embed_dim; idx += stride) {
            grad_b_O[idx]  = 0.0f;
            grad_b_Q[idx]  = 0.0f;
            grad_b_UK[idx] = 0.0f;
            grad_b_UV[idx] = 0.0f;
        }


        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < latent_dim; idx += stride) {
            grad_b_DKV[idx] = 0.0f;
        }

        __syncthreads();

        extern __shared__ float s_bias[];
        float *sbO  = s_bias;
        float *sbQ  = sbO  + embed_dim;
        float *sbUK = sbQ  + embed_dim;
        float *sbUV = sbUK + embed_dim;
        float *sbDKV= sbUV + embed_dim;

        for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
            sbO[i] = 0.0f; sbQ[i] = 0.0f;
            sbUK[i] = 0.0f; sbUV[i] = 0.0f;
        }
        for (int i = threadIdx.x; i < latent_dim; i += blockDim.x) {
            sbDKV[i] = 0.0f;
        }
        __syncthreads();

        for (int token = blockIdx.x * blockDim.x + threadIdx.x;
             token < total_tokens;
             token += stride) {
            int b = token / seq_len;
            int t = token % seq_len;

            // dOut -> gW_O, gB_O, dConcat
            for (int o = 0; o < embed_dim; ++o) {
                float go = grad_output[idx_bt(b, t, o, seq_len, embed_dim)];
                atomicAdd(&sbO[o], go);
                for (int i = 0; i < embed_dim; ++i) {
                    atomicAdd(&grad_W_O[o * embed_dim + i], go * concat[idx_bt(b, t, i, seq_len, embed_dim)]);
                    dConcat[idx_bt(b, t, i, seq_len, embed_dim)] += go * W_O[o * embed_dim + i];
                }
            }

            // back through attention: dConcat -> dV, dAttn
            for (int h = 0; h < num_heads; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    float gc = dConcat[idx_bthd(b, t, h, d, seq_len, embed_dim, head_dim)];
                    for (int t2 = 0; t2 < seq_len; ++t2) {
                        float p = attn_probs[idx_bhtt(b, h, t, t2, seq_len, num_heads)];
                        atomicAdd(&dV[idx_bthd(b, t2, h, d, seq_len, embed_dim, head_dim)], p * gc);
                        atomicAdd(&dAttn[idx_bhtt(b, h, t, t2, seq_len, num_heads)],
                                  gc * v[idx_bthd(b, t2, h, d, seq_len, embed_dim, head_dim)]);
                    }
                }
            }

            // softmax backward -> dQ, dK
            for (int h = 0; h < num_heads; ++h) {
                float sum = 0.0f;
                for (int t2 = 0; t2 < seq_len; ++t2) {
                    int id = idx_bhtt(b, h, t, t2, seq_len, num_heads);
                    sum += dAttn[id] * attn_probs[id];
                }
                for (int t2 = 0; t2 < seq_len; ++t2) {
                    int id = idx_bhtt(b, h, t, t2, seq_len, num_heads);
                    float grad = attn_probs[id] * (dAttn[id] - sum);
                    grad *= 1.0f / sqrtf(static_cast<float>(head_dim));
                    for (int d = 0; d < head_dim; ++d) {
                        dQ[idx_bthd(b, t, h, d, seq_len, embed_dim, head_dim)] +=
                            grad * k[idx_bthd(b, t2, h, d, seq_len, embed_dim, head_dim)];
                        atomicAdd(&dK[idx_bthd(b, t2, h, d, seq_len, embed_dim, head_dim)],
                                  grad * q[idx_bthd(b, t, h, d, seq_len, embed_dim, head_dim)]);
                    }
                }
            }
            // dQ -> gW_Q,gB_Q,dX
            for (int o = 0; o < embed_dim; ++o) {
                float gq = dQ[idx_bt(b, t, o, seq_len, embed_dim)];
                atomicAdd(&sbQ[o], gq);
                for (int i = 0; i < embed_dim; ++i) {
                    float x = input[idx_bt(b, t, i, seq_len, embed_dim)];
                    atomicAdd(&grad_W_Q[o * embed_dim + i], gq * x);
                    grad_input[idx_bt(b, t, i, seq_len, embed_dim)] +=
                        gq * W_Q[o * embed_dim + i];
                }
            }
            // dK,dV -> dC
            for (int i = 0; i < latent_dim; ++i) {
                float s = 0.0f;
                for (int o = 0; o < embed_dim; ++o) {
                    s += W_UK[o * latent_dim + i] *
                         dK[idx_bt(b, t, o, seq_len, embed_dim)];
                    s += W_UV[o * latent_dim + i] *
                         dV[idx_bt(b, t, o, seq_len, embed_dim)];
                }
                dC[(b * seq_len + t) * latent_dim + i] = s;
            }
            // accumulate gradients for UK/UV and biases
            for (int o = 0; o < embed_dim; ++o) {
                float dk = dK[idx_bt(b, t, o, seq_len, embed_dim)];
                float dv = dV[idx_bt(b, t, o, seq_len, embed_dim)];
                atomicAdd(&sbUK[o], dk);
                atomicAdd(&sbUV[o], dv);
                for (int i = 0; i < latent_dim; ++i) {
                    float c = c_kv[(b * seq_len + t) * latent_dim + i];
                    atomicAdd(&grad_W_UK[o * latent_dim + i], dk * c);
                    atomicAdd(&grad_W_UV[o * latent_dim + i], dv * c);
                }
            }
            // dC -> dX
            for (int i = 0; i < embed_dim; ++i) {
                float s = 0.0f;
                for (int oc = 0; oc < latent_dim; ++oc) {
                    s += W_DKV[oc * embed_dim + i] *
                         dC[(b * seq_len + t) * latent_dim + oc];
                }
                grad_input[idx_bt(b, t, i, seq_len, embed_dim)] += s;
            }
            // accumulate gradients for W_DKV/b_DKV
            for (int oc = 0; oc < latent_dim; ++oc) {
                float dc = dC[(b * seq_len + t) * latent_dim + oc];
                atomicAdd(&sbDKV[oc], dc);
                for (int i = 0; i < embed_dim; ++i) {
                    atomicAdd(&grad_W_DKV[oc * embed_dim + i], dc * input[idx_bt(b, t, i, seq_len, embed_dim)]);
                }
            }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
            atomicAdd(&grad_b_O[i], sbO[i]);
            atomicAdd(&grad_b_Q[i], sbQ[i]);
            atomicAdd(&grad_b_UK[i], sbUK[i]);
            atomicAdd(&grad_b_UV[i], sbUV[i]);
        }
        for (int i = threadIdx.x; i < latent_dim; i += blockDim.x) {
            atomicAdd(&grad_b_DKV[i], sbDKV[i]);
        }
    }

    void launchMLABackward(const float *input, const float *W_DKV,
                           const float *b_DKV, const float *W_UK, const float *b_UK,
                           const float *W_UV, const float *b_UV, const float *W_Q,
                           const float *b_Q, const float *W_O, const float *b_O,
                           const float *q, const float *k, const float *v,
                           const float *c_kv, const float *attn_probs,
                       const float *concat, const float *grad_output,
                       float *grad_input, float *grad_W_DKV, float *grad_b_DKV,
                       float *grad_W_UK, float *grad_b_UK, float *grad_W_UV,
                       float *grad_b_UV, float *grad_W_Q, float *grad_b_Q,
                       float *grad_W_O, float *grad_b_O, int batch_size,
                       int seq_len, int embed_dim, int num_heads,
                       int latent_dim, cudaStream_t stream) {
        initMLABackwardWorkspace(batch_size, seq_len, embed_dim, num_heads,
                                 latent_dim);

        int total_tokens = batch_size * seq_len;
        int block = 256;
        int grid  = (total_tokens + block - 1) / block;
        size_t shmem = (4 * embed_dim + latent_dim) * sizeof(float);

        MLABackwardKernel<<<grid, block, shmem, stream>>>(input, W_DKV, b_DKV, W_UK, b_UK, W_UV, b_UV, W_Q, b_Q,
                                                          W_O, b_O, q, k, v, c_kv, attn_probs, concat,
                                                          grad_output, grad_input, grad_W_DKV, grad_b_DKV, grad_W_UK, grad_b_UK,
                                                          grad_W_UV, grad_b_UV, grad_W_Q, grad_b_Q, grad_W_O, grad_b_O,
                                                          g_dConcat, g_dAttn, g_dV, g_dK, g_dQ, g_dC,
                                                          batch_size, seq_len, embed_dim, num_heads, latent_dim);

        CUDA_CHECK(cudaGetLastError());
    }
} // namespace cuda::attentions

