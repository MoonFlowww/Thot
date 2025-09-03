#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include "../../cuh/attentions/mha.cuh"

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); } } while(0)
#endif

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
        cudaStream_t /*stream*/) {

        int head_dim = embed_dim / num_heads;
        size_t BS = static_cast<size_t>(batch_size) * seq_len;
        size_t input_elems = BS * embed_dim;
        size_t weight_elems = static_cast<size_t>(embed_dim) * embed_dim;
        size_t attn_elems = static_cast<size_t>(batch_size) * num_heads * seq_len * seq_len;

        std::vector<float> h_input(input_elems);
        std::vector<float> h_Wq(weight_elems), h_Wk(weight_elems), h_Wv(weight_elems), h_Wo(weight_elems);
        CUDA_CHECK(cudaMemcpy(h_input.data(), input, input_elems * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_Wq.data(), Wq, weight_elems * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_Wk.data(), Wk, weight_elems * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_Wv.data(), Wv, weight_elems * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_Wo.data(), Wo, weight_elems * sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<float> h_Q(input_elems), h_K(input_elems), h_V(input_elems);
        std::vector<float> h_attn(attn_elems);
        std::vector<float> h_context(input_elems);
        std::vector<float> h_output(input_elems);

        for (size_t b = 0; b < BS; ++b) {
            for (int d2 = 0; d2 < embed_dim; ++d2) {
                float sumq = 0.0f, sumk = 0.0f, sumv = 0.0f;
                for (int d = 0; d < embed_dim; ++d) {
                    float x = h_input[b * embed_dim + d];
                    sumq += x * h_Wq[d * embed_dim + d2];
                    sumk += x * h_Wk[d * embed_dim + d2];
                    sumv += x * h_Wv[d * embed_dim + d2];
                }
                h_Q[b * embed_dim + d2] = sumq;
                h_K[b * embed_dim + d2] = sumk;
                h_V[b * embed_dim + d2] = sumv;
            }
        }

        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < num_heads; ++h) {
                for (int i = 0; i < seq_len; ++i) {
                    float* scores = &h_attn[((static_cast<size_t>(b) * num_heads + h) * seq_len + i) * seq_len];
                    for (int j = 0; j < seq_len; ++j) {
                        float s = 0.0f;
                        for (int k = 0; k < head_dim; ++k) {
                            float q = h_Q[(static_cast<size_t>(b) * seq_len + i) * embed_dim + h * head_dim + k];
                            float kv = h_K[(static_cast<size_t>(b) * seq_len + j) * embed_dim + h * head_dim + k];
                            s += q * kv;
                        }
                        scores[j] = s * scale;
                    }
                    float maxv = scores[0];
                    for (int j = 1; j < seq_len; ++j) maxv = std::max(maxv, scores[j]);
                    float sum = 0.0f;
                    for (int j = 0; j < seq_len; ++j) {
                        scores[j] = std::exp(scores[j] - maxv);
                        sum += scores[j];
                    }
                    for (int j = 0; j < seq_len; ++j) scores[j] /= sum;
                    for (int k = 0; k < head_dim; ++k) {
                        float val = 0.0f;
                        for (int j = 0; j < seq_len; ++j) {
                            float v = h_V[(static_cast<size_t>(b) * seq_len + j) * embed_dim + h * head_dim + k];
                            val += scores[j] * v;
                        }
                        h_context[(static_cast<size_t>(b) * seq_len + i) * embed_dim + h * head_dim + k] = val;
                    }
                }
            }
        }

        for (size_t b = 0; b < BS; ++b) {
            for (int d2 = 0; d2 < embed_dim; ++d2) {
                float sum = 0.0f;
                for (int d = 0; d < embed_dim; ++d) {
                    sum += h_context[b * embed_dim + d] * h_Wo[d * embed_dim + d2];
                }
                h_output[b * embed_dim + d2] = sum;
            }
        }

        CUDA_CHECK(cudaMemcpy(output, h_output.data(), input_elems * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(q_cache, h_Q.data(), input_elems * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(k_cache, h_K.data(), input_elems * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(v_cache, h_V.data(), input_elems * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(attn_cache, h_attn.data(), attn_elems * sizeof(float), cudaMemcpyHostToDevice));
        if (context_cache) CUDA_CHECK(cudaMemcpy(context_cache, h_context.data(), input_elems * sizeof(float), cudaMemcpyHostToDevice));
    }

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
        cudaStream_t /*stream*/) {

        int head_dim = embed_dim / num_heads;
        size_t BS = static_cast<size_t>(batch_size) * seq_len;
        size_t input_elems = BS * embed_dim;
        size_t weight_elems = static_cast<size_t>(embed_dim) * embed_dim;
        size_t attn_elems = static_cast<size_t>(batch_size) * num_heads * seq_len * seq_len;

        std::vector<float> h_grad_output(input_elems);
        std::vector<float> h_input(input_elems);
        std::vector<float> h_Wq(weight_elems), h_Wk(weight_elems), h_Wv(weight_elems), h_Wo(weight_elems);
        std::vector<float> h_Q(input_elems), h_K(input_elems), h_V(input_elems);
        std::vector<float> h_attn(attn_elems);
        std::vector<float> h_context(input_elems);

        CUDA_CHECK(cudaMemcpy(h_grad_output.data(), grad_output, input_elems * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_input.data(), input, input_elems * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_Wq.data(), Wq, weight_elems * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_Wk.data(), Wk, weight_elems * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_Wv.data(), Wv, weight_elems * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_Wo.data(), Wo, weight_elems * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_Q.data(), q_cache, input_elems * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_K.data(), k_cache, input_elems * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_V.data(), v_cache, input_elems * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_attn.data(), attn_cache, attn_elems * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_context.data(), context_cache, input_elems * sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<float> h_grad_input(input_elems, 0.0f);
        std::vector<float> h_grad_Wq(weight_elems, 0.0f);
        std::vector<float> h_grad_Wk(weight_elems, 0.0f);
        std::vector<float> h_grad_Wv(weight_elems, 0.0f);
        std::vector<float> h_grad_Wo(weight_elems, 0.0f);
        std::vector<float> h_grad_Q(input_elems, 0.0f);
        std::vector<float> h_grad_K(input_elems, 0.0f);
        std::vector<float> h_grad_V(input_elems, 0.0f);
        std::vector<float> h_grad_context(input_elems, 0.0f);

        for (int d = 0; d < embed_dim; ++d) {
            for (int d2 = 0; d2 < embed_dim; ++d2) {
                float sum = 0.0f;
                for (size_t b = 0; b < BS; ++b) {
                    sum += h_context[b * embed_dim + d] * h_grad_output[b * embed_dim + d2];
                }
                h_grad_Wo[d * embed_dim + d2] = sum / static_cast<float>(BS);
            }
        }

        for (size_t b = 0; b < BS; ++b) {
            for (int d = 0; d < embed_dim; ++d) {
                float sum = 0.0f;
                for (int d2 = 0; d2 < embed_dim; ++d2) {
                    sum += h_grad_output[b * embed_dim + d2] * h_Wo[d * embed_dim + d2];
                }
                h_grad_context[b * embed_dim + d] = sum;
            }
        }

        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < num_heads; ++h) {
                for (int i = 0; i < seq_len; ++i) {
                    float* scores = &h_attn[((static_cast<size_t>(b) * num_heads + h) * seq_len + i) * seq_len];
                    std::vector<float> grad_attn(seq_len, 0.0f);
                    for (int j = 0; j < seq_len; ++j) {
                        float tmp = 0.0f;
                        for (int k = 0; k < head_dim; ++k) {
                            tmp += h_grad_context[(static_cast<size_t>(b) * seq_len + i) * embed_dim + h * head_dim + k] *
                                   h_V[(static_cast<size_t>(b) * seq_len + j) * embed_dim + h * head_dim + k];
                            h_grad_V[(static_cast<size_t>(b) * seq_len + j) * embed_dim + h * head_dim + k] +=
                                scores[j] * h_grad_context[(static_cast<size_t>(b) * seq_len + i) * embed_dim + h * head_dim + k];
                        }
                        grad_attn[j] = tmp;
                    }
                    float sum_g = 0.0f;
                    for (int j = 0; j < seq_len; ++j) sum_g += grad_attn[j] * scores[j];
                    std::vector<float> grad_scores(seq_len, 0.0f);
                    for (int j = 0; j < seq_len; ++j) {
                        grad_scores[j] = (grad_attn[j] - sum_g) * scores[j];
                    }
                    for (int j = 0; j < seq_len; ++j) {
                        float gs = grad_scores[j] * scale;
                        for (int k = 0; k < head_dim; ++k) {
                            float qv = h_Q[(static_cast<size_t>(b) * seq_len + i) * embed_dim + h * head_dim + k];
                            float kv = h_K[(static_cast<size_t>(b) * seq_len + j) * embed_dim + h * head_dim + k];
                            h_grad_Q[(static_cast<size_t>(b) * seq_len + i) * embed_dim + h * head_dim + k] += gs * kv;
                            h_grad_K[(static_cast<size_t>(b) * seq_len + j) * embed_dim + h * head_dim + k] += gs * qv;
                        }
                    }
                }
            }
        }

        for (int d = 0; d < embed_dim; ++d) {
            for (int d2 = 0; d2 < embed_dim; ++d2) {
                float sumq = 0.0f, sumk = 0.0f, sumv = 0.0f;
                for (size_t b = 0; b < BS; ++b) {
                    float x = h_input[b * embed_dim + d];
                    sumq += x * h_grad_Q[b * embed_dim + d2];
                    sumk += x * h_grad_K[b * embed_dim + d2];
                    sumv += x * h_grad_V[b * embed_dim + d2];
                }
                h_grad_Wq[d * embed_dim + d2] = sumq / static_cast<float>(BS);
                h_grad_Wk[d * embed_dim + d2] = sumk / static_cast<float>(BS);
                h_grad_Wv[d * embed_dim + d2] = sumv / static_cast<float>(BS);
            }
        }

        for (size_t b = 0; b < BS; ++b) {
            for (int d = 0; d < embed_dim; ++d) {
                float sum = 0.0f;
                for (int d2 = 0; d2 < embed_dim; ++d2) {
                    sum += h_grad_Q[b * embed_dim + d2] * h_Wq[d * embed_dim + d2];
                    sum += h_grad_K[b * embed_dim + d2] * h_Wk[d * embed_dim + d2];
                    sum += h_grad_V[b * embed_dim + d2] * h_Wv[d * embed_dim + d2];
                }
                h_grad_input[b * embed_dim + d] = sum;
            }
        }

        CUDA_CHECK(cudaMemcpy(grad_input, h_grad_input.data(), input_elems * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(grad_Wq, h_grad_Wq.data(), weight_elems * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(grad_Wk, h_grad_Wk.data(), weight_elems * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(grad_Wv, h_grad_Wv.data(), weight_elems * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(grad_Wo, h_grad_Wo.data(), weight_elems * sizeof(float), cudaMemcpyHostToDevice));
    }

} // namespace cuda::attentions