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

namespace {
    inline int idx_bt(int b, int t, int e, int seq, int embed) {
        return (b * seq + t) * embed + e;
    }
    inline int idx_bthd(int b, int t, int h, int d, int seq, int embed, int head_dim) {
        return (b * seq + t) * embed + h * head_dim + d;
    }
    inline int idx_bhtt(int b, int h, int t, int t2, int seq, int num_heads) {
        return ((b * num_heads + h) * seq + t) * seq + t2;
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
                      cudaStream_t) {
    const int head_dim = embed_dim / num_heads;

    size_t X_size = static_cast<size_t>(batch_size) * seq_len * embed_dim;
    size_t C_size = static_cast<size_t>(batch_size) * seq_len * latent_dim;
    size_t prob_sz = static_cast<size_t>(batch_size) * num_heads * seq_len * seq_len;

    std::vector<float> X_h(X_size);
    std::vector<float> q_h(X_size, 0.0f), k_h(X_size, 0.0f), v_h(X_size, 0.0f);
    std::vector<float> C_h(C_size, 0.0f);
    std::vector<float> attn_h(prob_sz, 0.0f);
    std::vector<float> concat_h(X_size, 0.0f);
    std::vector<float> out_h(X_size, 0.0f);

    CUDA_CHECK(cudaMemcpy(X_h.data(), input, X_size * sizeof(float), cudaMemcpyDeviceToHost));

    // C = W_DKV * X + b_DKV
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            for (int oc = 0; oc < latent_dim; ++oc) {
                float s = b_DKV ? b_DKV[oc] : 0.0f;
                for (int i = 0; i < embed_dim; ++i) {
                    s += W_DKV[oc * embed_dim + i] * X_h[idx_bt(b, t, i, seq_len, embed_dim)];
                }
                C_h[(b * seq_len + t) * latent_dim + oc] = s;
            }
        }
    }

    // Q = W_Q * X + b_Q
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            for (int o = 0; o < embed_dim; ++o) {
                float s = b_Q ? b_Q[o] : 0.0f;
                for (int i = 0; i < embed_dim; ++i) {
                    s += W_Q[o * embed_dim + i] * X_h[idx_bt(b, t, i, seq_len, embed_dim)];
                }
                q_h[idx_bt(b, t, o, seq_len, embed_dim)] = s;
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
                    float c = C_h[(b * seq_len + t) * latent_dim + i];
                    sk += W_UK[o * latent_dim + i] * c;
                    sv += W_UV[o * latent_dim + i] * c;
                }
                k_h[idx_bt(b, t, o, seq_len, embed_dim)] = sk;
                v_h[idx_bt(b, t, o, seq_len, embed_dim)] = sv;
            }
        }
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Attention
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int t = 0; t < seq_len; ++t) {
                std::vector<float> scores(seq_len);
                for (int t2 = 0; t2 < seq_len; ++t2) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += q_h[idx_bthd(b, t, h, d, seq_len, embed_dim, head_dim)] *
                               k_h[idx_bthd(b, t2, h, d, seq_len, embed_dim, head_dim)];
                    }
                    scores[t2] = dot * scale;
                }
                float mx = scores[0];
                for (int t2 = 1; t2 < seq_len; ++t2) mx = std::max(mx, scores[t2]);
                float sum = 0.0f;
                for (int t2 = 0; t2 < seq_len; ++t2) {
                    float e = std::exp(scores[t2] - mx);
                    attn_h[idx_bhtt(b, h, t, t2, seq_len, num_heads)] = e;
                    sum += e;
                }
                for (int t2 = 0; t2 < seq_len; ++t2) {
                    attn_h[idx_bhtt(b, h, t, t2, seq_len, num_heads)] /= sum;
                }
                for (int d = 0; d < head_dim; ++d) {
                    float acc = 0.0f;
                    for (int t2 = 0; t2 < seq_len; ++t2) {
                        float p = attn_h[idx_bhtt(b, h, t, t2, seq_len, num_heads)];
                        acc += p * v_h[idx_bthd(b, t2, h, d, seq_len, embed_dim, head_dim)];
                    }
                    concat_h[idx_bthd(b, t, h, d, seq_len, embed_dim, head_dim)] = acc;
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
                    s += W_O[o * embed_dim + i] * concat_h[idx_bt(b, t, i, seq_len, embed_dim)];
                }
                out_h[idx_bt(b, t, o, seq_len, embed_dim)] = s;
            }
        }
    }

    CUDA_CHECK(cudaMemcpy(output, out_h.data(), X_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(q, q_h.data(), X_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(k, k_h.data(), X_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(v, v_h.data(), X_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(attn_probs, attn_h.data(), prob_sz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(concat, concat_h.data(), X_size * sizeof(float), cudaMemcpyHostToDevice));
    if (c_kv) CUDA_CHECK(cudaMemcpy(c_kv, C_h.data(), C_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaGetLastError());
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
                       cudaStream_t) {
    const int head_dim = embed_dim / num_heads;

    size_t X_size = static_cast<size_t>(batch_size) * seq_len * embed_dim;
    size_t C_size = static_cast<size_t>(batch_size) * seq_len * latent_dim;
    size_t prob_sz = static_cast<size_t>(batch_size) * num_heads * seq_len * seq_len;

    std::vector<float> X_h(X_size);
    std::vector<float> q_h(X_size), k_h(X_size), v_h(X_size);
    std::vector<float> C_h(C_size);
    std::vector<float> attn_h(prob_sz), concat_h(X_size);
    std::vector<float> dOut_h(X_size);

    CUDA_CHECK(cudaMemcpy(X_h.data(), input, X_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(q_h.data(), q, X_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(k_h.data(), k, X_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_h.data(), v, X_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(attn_h.data(), attn_probs, prob_sz * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(concat_h.data(), concat, X_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dOut_h.data(), grad_output, X_size * sizeof(float), cudaMemcpyDeviceToHost));
    if (c_kv) CUDA_CHECK(cudaMemcpy(C_h.data(), c_kv, C_size * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> dConcat_h(X_size, 0.0f);
    std::vector<float> dAttn_h(prob_sz, 0.0f);
    std::vector<float> dV_h(X_size, 0.0f);
    std::vector<float> dK_h(X_size, 0.0f);
    std::vector<float> dQ_h(X_size, 0.0f);
    std::vector<float> dC_h(C_size, 0.0f);

    std::vector<float> gW_O(embed_dim * embed_dim, 0.0f);
    std::vector<float> gB_O(embed_dim, 0.0f);
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            for (int o = 0; o < embed_dim; ++o) {
                float go = dOut_h[idx_bt(b, t, o, seq_len, embed_dim)];
                gB_O[o] += go;
                for (int i = 0; i < embed_dim; ++i) {
                    gW_O[o * embed_dim + i] += go * concat_h[idx_bt(b, t, i, seq_len, embed_dim)];
                    dConcat_h[idx_bt(b, t, i, seq_len, embed_dim)] += go * W_O[o * embed_dim + i];
                }
            }
        }
    }

    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int t = 0; t < seq_len; ++t) {
                for (int d = 0; d < head_dim; ++d) {
                    float gc = dConcat_h[idx_bthd(b, t, h, d, seq_len, embed_dim, head_dim)];
                    for (int t2 = 0; t2 < seq_len; ++t2) {
                        float p = attn_h[idx_bhtt(b, h, t, t2, seq_len, num_heads)];
                        dV_h[idx_bthd(b, t2, h, d, seq_len, embed_dim, head_dim)] += p * gc;
                        dAttn_h[idx_bhtt(b, h, t, t2, seq_len, num_heads)] += gc * v_h[idx_bthd(b, t2, h, d, seq_len, embed_dim, head_dim)];
                    }
                }
            }
        }
    }

    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int t = 0; t < seq_len; ++t) {
                float sum = 0.0f;
                for (int t2 = 0; t2 < seq_len; ++t2) {
                    int id = idx_bhtt(b, h, t, t2, seq_len, num_heads);
                    sum += dAttn_h[id] * attn_h[id];
                }
                for (int t2 = 0; t2 < seq_len; ++t2) {
                    int id = idx_bhtt(b, h, t, t2, seq_len, num_heads);
                    float grad = attn_h[id] * (dAttn_h[id] - sum);
                    grad *= 1.0f / std::sqrt(static_cast<float>(head_dim));
                    for (int d = 0; d < head_dim; ++d) {
                        dQ_h[idx_bthd(b, t, h, d, seq_len, embed_dim, head_dim)] += grad * k_h[idx_bthd(b, t2, h, d, seq_len, embed_dim, head_dim)];
                        dK_h[idx_bthd(b, t2, h, d, seq_len, embed_dim, head_dim)] += grad * q_h[idx_bthd(b, t, h, d, seq_len, embed_dim, head_dim)];
                    }
                }
            }
        }
    }

    std::vector<float> gW_Q(embed_dim * embed_dim, 0.0f);
    std::vector<float> gB_Q(embed_dim, 0.0f);
    std::vector<float> dX_h(X_size, 0.0f);
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            for (int o = 0; o < embed_dim; ++o) {
                float gq = dQ_h[idx_bt(b, t, o, seq_len, embed_dim)];
                gB_Q[o] += gq;
                for (int i = 0; i < embed_dim; ++i) {
                    float x = X_h[idx_bt(b, t, i, seq_len, embed_dim)];
                    gW_Q[o * embed_dim + i] += gq * x;
                    dX_h[idx_bt(b, t, i, seq_len, embed_dim)] += gq * W_Q[o * embed_dim + i];
                }
            }
        }
    }

    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            for (int i = 0; i < latent_dim; ++i) {
                float s = 0.0f;
                for (int o = 0; o < embed_dim; ++o) {
                    s += W_UK[o * latent_dim + i] * dK_h[idx_bt(b, t, o, seq_len, embed_dim)];
                    s += W_UV[o * latent_dim + i] * dV_h[idx_bt(b, t, o, seq_len, embed_dim)];
                }
                dC_h[(b * seq_len + t) * latent_dim + i] = s;
            }
        }
    }

    std::vector<float> gW_UK(embed_dim * latent_dim, 0.0f);
    std::vector<float> gB_UK(embed_dim, 0.0f);
    std::vector<float> gW_UV(embed_dim * latent_dim, 0.0f);
    std::vector<float> gB_UV(embed_dim, 0.0f);
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            for (int o = 0; o < embed_dim; ++o) {
                float dk = dK_h[idx_bt(b, t, o, seq_len, embed_dim)];
                float dv = dV_h[idx_bt(b, t, o, seq_len, embed_dim)];
                gB_UK[o] += dk;
                gB_UV[o] += dv;
                for (int i = 0; i < latent_dim; ++i) {
                    float c = C_h[(b * seq_len + t) * latent_dim + i];
                    gW_UK[o * latent_dim + i] += dk * c;
                    gW_UV[o * latent_dim + i] += dv * c;
                }
            }
        }
    }

    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            for (int i = 0; i < embed_dim; ++i) {
                float s = 0.0f;
                for (int oc = 0; oc < latent_dim; ++oc) {
                    s += W_DKV[oc * embed_dim + i] * dC_h[(b * seq_len + t) * latent_dim + oc];
                }
                dX_h[idx_bt(b, t, i, seq_len, embed_dim)] += s;
            }
        }
    }

    std::vector<float> gW_DKV(latent_dim * embed_dim, 0.0f);
    std::vector<float> gB_DKV(latent_dim, 0.0f);
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            for (int oc = 0; oc < latent_dim; ++oc) {
                float dc = dC_h[(b * seq_len + t) * latent_dim + oc];
                gB_DKV[oc] += dc;
                for (int i = 0; i < embed_dim; ++i) {
                    gW_DKV[oc * embed_dim + i] += dc * X_h[idx_bt(b, t, i, seq_len, embed_dim)];
                }
            }
        }
    }

    CUDA_CHECK(cudaMemcpy(grad_input, dX_h.data(), X_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grad_W_DKV, gW_DKV.data(), gW_DKV.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grad_b_DKV, gB_DKV.data(), gB_DKV.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grad_W_UK, gW_UK.data(), gW_UK.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grad_b_UK, gB_UK.data(), gB_UK.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grad_W_UV, gW_UV.data(), gW_UV.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grad_b_UV, gB_UV.data(), gB_UV.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grad_W_Q, gW_Q.data(), gW_Q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grad_b_Q, gB_Q.data(), gB_Q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grad_W_O, gW_O.data(), gW_O.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grad_b_O, gB_O.data(), gB_O.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda::attentions
