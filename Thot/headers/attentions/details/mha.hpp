#pragma once

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include "../../tensor.hpp"
#include "../../initializations/initializations.hpp"
#include "../../../cuda/cuh/attentions/mha.cuh"
#include "../../layers/layers.hpp"

namespace Thot {

    // Multi-head attention layer implementation
    class MHAAttentionLayer : public Layer {
    private:
        int embed_dim_;
        int num_heads_;
        int head_dim_;
        Initialization initialization_;

        // weights and biases for projections
        Utils::Tensor W_q_;
        Utils::Tensor W_k_;
        Utils::Tensor W_v_;
        Utils::Tensor W_o_;

        Utils::Tensor b_q_;
        Utils::Tensor b_k_;
        Utils::Tensor b_v_;
        Utils::Tensor b_o_;

        // gradients
        Utils::Tensor grad_W_q_;
        Utils::Tensor grad_W_k_;
        Utils::Tensor grad_W_v_;
        Utils::Tensor grad_W_o_;

        Utils::Tensor grad_b_q_;
        Utils::Tensor grad_b_k_;
        Utils::Tensor grad_b_v_;
        Utils::Tensor grad_b_o_;

        // caches for backward
        Utils::Tensor q_;
        Utils::Tensor k_;
        Utils::Tensor v_;
        Utils::Tensor attn_probs_;
        Utils::Tensor concat_;

    public:
        MHAAttentionLayer(int embed_dim, int num_heads, Initialization init,
                          const std::string &name = "MHA")
            : Layer(name), embed_dim_(embed_dim), num_heads_(num_heads),
              initialization_(init) {
            if (embed_dim_ % num_heads_ != 0) {
                throw std::invalid_argument("embed_dim must be divisible by num_heads");
            }
            head_dim_ = embed_dim_ / num_heads_;

            W_q_ = Utils::Tensor({embed_dim_, embed_dim_});
            W_k_ = Utils::Tensor({embed_dim_, embed_dim_});
            W_v_ = Utils::Tensor({embed_dim_, embed_dim_});
            W_o_ = Utils::Tensor({embed_dim_, embed_dim_});

            b_q_ = Utils::Tensor({embed_dim_});
            b_k_ = Utils::Tensor({embed_dim_});
            b_v_ = Utils::Tensor({embed_dim_});
            b_o_ = Utils::Tensor({embed_dim_});

            grad_W_q_ = Utils::Tensor({embed_dim_, embed_dim_}, true);
            grad_W_k_ = Utils::Tensor({embed_dim_, embed_dim_}, true);
            grad_W_v_ = Utils::Tensor({embed_dim_, embed_dim_}, true);
            grad_W_o_ = Utils::Tensor({embed_dim_, embed_dim_}, true);

            grad_b_q_ = Utils::Tensor({embed_dim_}, true);
            grad_b_k_ = Utils::Tensor({embed_dim_}, true);
            grad_b_v_ = Utils::Tensor({embed_dim_}, true);
            grad_b_o_ = Utils::Tensor({embed_dim_}, true);

            Initializations::initialize_tensor(W_q_, init, embed_dim_, embed_dim_);
            Initializations::initialize_tensor(W_k_, init, embed_dim_, embed_dim_);
            Initializations::initialize_tensor(W_v_, init, embed_dim_, embed_dim_);
            Initializations::initialize_tensor(W_o_, init, embed_dim_, embed_dim_);

            Initializations::zeros(b_q_);
            Initializations::zeros(b_k_);
            Initializations::zeros(b_v_);
            Initializations::zeros(b_o_);
        }

        Utils::Tensor forward(const Utils::Tensor &input) override {
            this->input_cache_ = Utils::Tensor(input.shape());
            float *src_ptr = static_cast<float *>(input.data());
            float *dst_ptr = static_cast<float *>(this->input_cache_.data());
            size_t bytes = input.size() * sizeof(float);
            ::cudaMemcpy(dst_ptr, src_ptr, bytes, ::cudaMemcpyDeviceToDevice);

            int batch = input.shape()[0];
            int seq = input.shape().size() > 1 ? input.shape()[1] : 1;

            Utils::Tensor output({batch, seq, embed_dim_});
            q_ = Utils::Tensor({batch, seq, embed_dim_});
            k_ = Utils::Tensor({batch, seq, embed_dim_});
            v_ = Utils::Tensor({batch, seq, embed_dim_});
            attn_probs_ = Utils::Tensor({batch, num_heads_, seq, seq});
            concat_ = Utils::Tensor({batch, seq, embed_dim_});

            cuda::attentions::launchMHAForward(
                static_cast<const float *>(this->input_cache_.data()),
                static_cast<const float *>(W_q_.data()), static_cast<const float *>(b_q_.data()),
                static_cast<const float *>(W_k_.data()), static_cast<const float *>(b_k_.data()),
                static_cast<const float *>(W_v_.data()), static_cast<const float *>(b_v_.data()),
                static_cast<const float *>(W_o_.data()), static_cast<const float *>(b_o_.data()),
                static_cast<float *>(output.data()),
                static_cast<float *>(q_.data()), static_cast<float *>(k_.data()),
                static_cast<float *>(v_.data()), static_cast<float *>(attn_probs_.data()),
                static_cast<float *>(concat_.data()),
                batch, seq, embed_dim_, num_heads_);

            return output;
        }

        Utils::Tensor backward(const Utils::Tensor &grad_output) override {
            int batch = grad_output.shape()[0];
            int seq = grad_output.shape().size() > 1 ? grad_output.shape()[1] : 1;

            Utils::Tensor grad_input({batch, seq, embed_dim_});

            cuda::attentions::launchMHABackward(
                static_cast<const float *>(this->input_cache_.data()),
                static_cast<const float *>(W_q_.data()), static_cast<const float *>(b_q_.data()),
                static_cast<const float *>(W_k_.data()), static_cast<const float *>(b_k_.data()),
                static_cast<const float *>(W_v_.data()), static_cast<const float *>(b_v_.data()),
                static_cast<const float *>(W_o_.data()), static_cast<const float *>(b_o_.data()),
                static_cast<const float *>(q_.data()), static_cast<const float *>(k_.data()),
                static_cast<const float *>(v_.data()), static_cast<const float *>(attn_probs_.data()),
                static_cast<const float *>(concat_.data()),
                static_cast<const float *>(grad_output.data()),
                static_cast<float *>(grad_input.data()),
                static_cast<float *>(grad_W_q_.data()), static_cast<float *>(grad_b_q_.data()),
                static_cast<float *>(grad_W_k_.data()), static_cast<float *>(grad_b_k_.data()),
                static_cast<float *>(grad_W_v_.data()), static_cast<float *>(grad_b_v_.data()),
                static_cast<float *>(grad_W_o_.data()), static_cast<float *>(grad_b_o_.data()),
                batch, seq, embed_dim_, num_heads_);

            if (this->optimizer_) {
                this->optimizer_->update(W_q_, grad_W_q_);
                this->optimizer_->update(W_k_, grad_W_k_);
                this->optimizer_->update(W_v_, grad_W_v_);
                this->optimizer_->update(W_o_, grad_W_o_);
                this->optimizer_->update(b_q_, grad_b_q_);
                this->optimizer_->update(b_k_, grad_b_k_);
                this->optimizer_->update(b_v_, grad_b_v_);
                this->optimizer_->update(b_o_, grad_b_o_);
            }

            return grad_input;
        }

        size_t get_flops(int batch_size = 1) const override {
            return static_cast<size_t>(batch_size) * 4 * embed_dim_ * embed_dim_;
        }

        size_t get_parameters() const override {
            return 4 * embed_dim_ * embed_dim_ + 4 * embed_dim_;
        }
        Utils::Tensor& W_q() { return W_q_; }
        Utils::Tensor& W_k() { return W_k_; }

        Utils::Tensor& b_q() { return b_q_; }
        Utils::Tensor& b_k() { return b_k_; }

        Utils::Tensor& b_v() { return b_v_; }
        Utils::Tensor& b_o() { return b_o_; }

        Utils::Tensor& W_v() { return W_v_; }
        Utils::Tensor& W_o() { return W_o_; }

        Activation get_activation() const override { return Activation::Linear; }

        Initialization get_initialization() const override { return initialization_; }

        int get_input_size() const override { return embed_dim_; }

        int get_output_size() const override { return embed_dim_; }
    };
} // namespace Thot