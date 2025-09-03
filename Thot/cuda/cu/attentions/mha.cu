#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <algorithm>

#include "../../cuh/attentions/mha.cuh"

namespace cuda::attentions {

    namespace {
        inline int idx_bt(int b, int t, int e, int seq, int embed) {
            return (b * seq + t) * embed + e;
        }
        inline int idx_bthd(int b, int t, int h, int d, int seq, int embed, int head_dim) {
            return (b * seq + t) * embed + h * head_dim + d;
        }
        inline int idx_bhtt(int b, int h, int t1, int t2, int seq, int num_heads) {
            return ((b * num_heads + h) * seq + t1) * seq + t2;
        }
    }

    void launchMHAForward(const float *input,
                          const float *w_q, const float *b_q,
                          const float *w_k, const float *b_k,
                          const float *w_v, const float *b_v,
                          const float *w_o, const float *b_o,
                          float *output,
                          float *q, float *k, float *v,
                          float *attn_probs, float *concat,
                          int batch_size, int seq_len, int embed_dim, int num_heads,
                          cudaStream_t) {

        const int head_dim = embed_dim / num_heads;

        size_t input_size = static_cast<size_t>(batch_size) * seq_len * embed_dim;
        size_t weight_size = static_cast<size_t>(embed_dim) * embed_dim;
        size_t bias_size = static_cast<size_t>(embed_dim);
        size_t prob_size = static_cast<size_t>(batch_size) * num_heads * seq_len * seq_len;

        std::vector<float> in_h(input_size);
        std::vector<float> wq_h(weight_size), wk_h(weight_size), wv_h(weight_size), wo_h(weight_size);
        std::vector<float> bq_h(bias_size), bk_h(bias_size), bv_h(bias_size), bo_h(bias_size);

        cudaMemcpy(in_h.data(), input, input_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(wq_h.data(), w_q, weight_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(wk_h.data(), w_k, weight_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(wv_h.data(), w_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(wo_h.data(), w_o, weight_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(bq_h.data(), b_q, bias_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(bk_h.data(), b_k, bias_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(bv_h.data(), b_v, bias_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(bo_h.data(), b_o, bias_size * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<float> q_h(input_size, 0.0f), k_h(input_size, 0.0f), v_h(input_size, 0.0f);
        std::vector<float> attn_h(prob_size, 0.0f);
        std::vector<float> concat_h(input_size, 0.0f);
        std::vector<float> out_h(input_size, 0.0f);

        // Linear projections for Q, K, V
        for (int b = 0; b < batch_size; ++b) {
            for (int t = 0; t < seq_len; ++t) {
                for (int o = 0; o < embed_dim; ++o) {
                    float sq = bq_h[o];
                    float sk = bk_h[o];
                    float sv = bv_h[o];
                    for (int i = 0; i < embed_dim; ++i) {
                        float val = in_h[idx_bt(b, t, i, seq_len, embed_dim)];
                        sq += val * wq_h[i * embed_dim + o];
                        sk += val * wk_h[i * embed_dim + o];
                        sv += val * wv_h[i * embed_dim + o];
                    }
                    q_h[idx_bt(b, t, o, seq_len, embed_dim)] = sq;
                    k_h[idx_bt(b, t, o, seq_len, embed_dim)] = sk;
                    v_h[idx_bt(b, t, o, seq_len, embed_dim)] = sv;
                }
            }
        }

        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // Scaled dot-product attention per head
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
                    float max_score = scores[0];
                    for (int t2 = 1; t2 < seq_len; ++t2)
                        max_score = std::max(max_score, scores[t2]);
                    float sum_exp = 0.0f;
                    for (int t2 = 0; t2 < seq_len; ++t2) {
                        float e = std::exp(scores[t2] - max_score);
                        scores[t2] = e;
                        sum_exp += e;
                    }
                    for (int t2 = 0; t2 < seq_len; ++t2) {
                        float p = scores[t2] / sum_exp;
                        attn_h[idx_bhtt(b, h, t, t2, seq_len, num_heads)] = p;
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

        // Final output projection
        for (int b = 0; b < batch_size; ++b) {
            for (int t = 0; t < seq_len; ++t) {
                for (int o = 0; o < embed_dim; ++o) {
                    float sum = bo_h[o];
                    for (int i = 0; i < embed_dim; ++i) {
                        sum += concat_h[idx_bt(b, t, i, seq_len, embed_dim)] *
                               wo_h[i * embed_dim + o];
                    }
                    out_h[idx_bt(b, t, o, seq_len, embed_dim)] = sum;
                }
            }
        }

        cudaMemcpy(q, q_h.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(k, k_h.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(v, v_h.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(attn_probs, attn_h.data(), prob_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(concat, concat_h.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(output, out_h.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
    }

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
                           cudaStream_t) {

        const int head_dim = embed_dim / num_heads;
        size_t input_size = static_cast<size_t>(batch_size) * seq_len * embed_dim;
        size_t weight_size = static_cast<size_t>(embed_dim) * embed_dim;
        size_t bias_size = static_cast<size_t>(embed_dim);
        size_t prob_size = static_cast<size_t>(batch_size) * num_heads * seq_len * seq_len;

        std::vector<float> in_h(input_size);
        std::vector<float> wq_h(weight_size), wk_h(weight_size), wv_h(weight_size), wo_h(weight_size);
        std::vector<float> q_h(input_size), k_h(input_size), v_h(input_size);
        std::vector<float> attn_h(prob_size), concat_h(input_size);
        std::vector<float> grad_out_h(input_size);

        cudaMemcpy(in_h.data(), input, input_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(wq_h.data(), w_q, weight_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(wk_h.data(), w_k, weight_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(wv_h.data(), w_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(wo_h.data(), w_o, weight_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(q_h.data(), q, input_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(k_h.data(), k, input_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(v_h.data(), v, input_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(attn_h.data(), attn_probs, prob_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(concat_h.data(), concat, input_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(grad_out_h.data(), grad_output, input_size * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<float> grad_in_h(input_size, 0.0f);
        std::vector<float> grad_concat_h(input_size, 0.0f);
        std::vector<float> grad_q_h(input_size, 0.0f), grad_k_h(input_size, 0.0f), grad_v_h(input_size, 0.0f);
        std::vector<float> grad_probs_h(prob_size, 0.0f), grad_scores_h(prob_size, 0.0f);
        std::vector<float> grad_wo_h(weight_size, 0.0f), grad_bo_h(bias_size, 0.0f);
        std::vector<float> grad_wq_h(weight_size, 0.0f), grad_wk_h(weight_size, 0.0f), grad_wv_h(weight_size, 0.0f);
        std::vector<float> grad_bq_h(bias_size, 0.0f), grad_bk_h(bias_size, 0.0f), grad_bv_h(bias_size, 0.0f);

        // Output projection backward
        for (int b = 0; b < batch_size; ++b) {
            for (int t = 0; t < seq_len; ++t) {
                for (int o = 0; o < embed_dim; ++o) {
                    float go = grad_out_h[idx_bt(b, t, o, seq_len, embed_dim)];
                    grad_bo_h[o] += go;
                    for (int i = 0; i < embed_dim; ++i) {
                        grad_wo_h[i * embed_dim + o] += concat_h[idx_bt(b, t, i, seq_len, embed_dim)] * go;
                        grad_concat_h[idx_bt(b, t, i, seq_len, embed_dim)] += go * wo_h[i * embed_dim + o];
                    }
                }
            }
        }

        // Convert grad_concat to per-head structure
        std::vector<float> grad_context_h(batch_size * num_heads * seq_len * head_dim, 0.0f);
        for (int b = 0; b < batch_size; ++b)
            for (int t = 0; t < seq_len; ++t)
                for (int h = 0; h < num_heads; ++h)
                    for (int d = 0; d < head_dim; ++d)
                        grad_context_h[idx_bthd(b, t, h, d, seq_len, embed_dim, head_dim)] =
                            grad_concat_h[idx_bt(b, t, h * head_dim + d, seq_len, embed_dim)];

        // Gradients for V and attention probabilities
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < num_heads; ++h) {
                for (int t = 0; t < seq_len; ++t) {
                    for (int t2 = 0; t2 < seq_len; ++t2) {
                        for (int d = 0; d < head_dim; ++d) {
                            float gc = grad_context_h[idx_bthd(b, t, h, d, seq_len, embed_dim, head_dim)];
                            float vv = v_h[idx_bthd(b, t2, h, d, seq_len, embed_dim, head_dim)];
                            grad_probs_h[idx_bhtt(b, h, t, t2, seq_len, num_heads)] += gc * vv;
                            grad_v_h[idx_bthd(b, t2, h, d, seq_len, embed_dim, head_dim)] +=
                                attn_h[idx_bhtt(b, h, t, t2, seq_len, num_heads)] * gc;
                        }
                    }
                }
            }
        }

        // Softmax backward to scores
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < num_heads; ++h) {
                for (int t = 0; t < seq_len; ++t) {
                    float sum = 0.0f;
                    for (int t2 = 0; t2 < seq_len; ++t2) {
                        int id = idx_bhtt(b, h, t, t2, seq_len, num_heads);
                        sum += grad_probs_h[id] * attn_h[id];
                    }
                    for (int t2 = 0; t2 < seq_len; ++t2) {
                        int id = idx_bhtt(b, h, t, t2, seq_len, num_heads);
                        grad_scores_h[id] = (grad_probs_h[id] - sum) * attn_h[id];
                    }
                }
            }
        }

        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // Scores to Q and K
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < num_heads; ++h) {
                for (int t = 0; t < seq_len; ++t) {
                    for (int t2 = 0; t2 < seq_len; ++t2) {
                        float gs = grad_scores_h[idx_bhtt(b, h, t, t2, seq_len, num_heads)] * scale;
                        for (int d = 0; d < head_dim; ++d) {
                            grad_q_h[idx_bthd(b, t, h, d, seq_len, embed_dim, head_dim)] +=
                                gs * k_h[idx_bthd(b, t2, h, d, seq_len, embed_dim, head_dim)];
                            grad_k_h[idx_bthd(b, t2, h, d, seq_len, embed_dim, head_dim)] +=
                                gs * q_h[idx_bthd(b, t, h, d, seq_len, embed_dim, head_dim)];
                        }
                    }
                }
            }
        }

        // Gradients w.r.t. projection matrices and input
        for (int b = 0; b < batch_size; ++b) {
            for (int t = 0; t < seq_len; ++t) {
                for (int o = 0; o < embed_dim; ++o) {
                    float gq = grad_q_h[idx_bt(b, t, o, seq_len, embed_dim)];
                    float gk = grad_k_h[idx_bt(b, t, o, seq_len, embed_dim)];
                    float gv = grad_v_h[idx_bt(b, t, o, seq_len, embed_dim)];
                    grad_bq_h[o] += gq;
                    grad_bk_h[o] += gk;
                    grad_bv_h[o] += gv;
                    for (int i = 0; i < embed_dim; ++i) {
                        float in_val = in_h[idx_bt(b, t, i, seq_len, embed_dim)];
                        grad_wq_h[i * embed_dim + o] += in_val * gq;
                        grad_wk_h[i * embed_dim + o] += in_val * gk;
                        grad_wv_h[i * embed_dim + o] += in_val * gv;
                        grad_in_h[idx_bt(b, t, i, seq_len, embed_dim)] +=
                            gq * wq_h[i * embed_dim + o] +
                            gk * wk_h[i * embed_dim + o] +
                            gv * wv_h[i * embed_dim + o];
                    }
                }
            }
        }

        float norm = 1.0f / static_cast<float>(batch_size * seq_len);
        for (size_t i = 0; i < weight_size; ++i) {
            grad_wq_h[i] *= norm;
            grad_wk_h[i] *= norm;
            grad_wv_h[i] *= norm;
            grad_wo_h[i] *= norm;
        }
        for (size_t i = 0; i < bias_size; ++i) {
            grad_bq_h[i] *= norm;
            grad_bk_h[i] *= norm;
            grad_bv_h[i] *= norm;
            grad_bo_h[i] *= norm;
        }

        cudaMemcpy(grad_input, grad_in_h.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(grad_w_q, grad_wq_h.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(grad_w_k, grad_wk_h.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(grad_w_v, grad_wv_h.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(grad_w_o, grad_wo_h.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(grad_b_q, grad_bq_h.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(grad_b_k, grad_bk_h.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(grad_b_v, grad_bv_h.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(grad_b_o, grad_bo_h.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice);
    }

} // namespace cuda::attentions