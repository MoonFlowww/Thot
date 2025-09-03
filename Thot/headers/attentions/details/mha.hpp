#pragma once

#include "../../tensor.hpp"
#include "../../initializations/initializations.hpp"
#include "../../optimizations/optimizations.hpp"
#include "../../../cuda/cuh/attentions/mha.cuh"

namespace Thot {

class Attention;

class MHAAtt : public Attention {
private:
    int embed_dim_;
    int num_heads_;
    int seq_len_cache_;
    Initialization initialization_type_;

    Utils::Tensor Wq_;
    Utils::Tensor Wk_;
    Utils::Tensor Wv_;
    Utils::Tensor Wo_;

    Utils::Tensor grad_Wq_;
    Utils::Tensor grad_Wk_;
    Utils::Tensor grad_Wv_;
    Utils::Tensor grad_Wo_;

    Utils::Tensor q_cache_;
    Utils::Tensor k_cache_;
    Utils::Tensor v_cache_;
    Utils::Tensor attn_cache_;
    Utils::Tensor context_cache_;

public:
    MHAAtt(int embed_dim, int num_heads, Initialization weight_init = Initialization::Xavier, const std::string& name = "MHA")
        : Attention(name), embed_dim_(embed_dim), num_heads_(num_heads), seq_len_cache_(0), initialization_type_(weight_init) {
        Wq_ = Utils::Tensor({ embed_dim_, embed_dim_ });
        Wk_ = Utils::Tensor({ embed_dim_, embed_dim_ });
        Wv_ = Utils::Tensor({ embed_dim_, embed_dim_ });
        Wo_ = Utils::Tensor({ embed_dim_, embed_dim_ });

        Initializers::initialize_tensor(Wq_, weight_init, embed_dim_, embed_dim_);
        Initializers::initialize_tensor(Wk_, weight_init, embed_dim_, embed_dim_);
        Initializers::initialize_tensor(Wv_, weight_init, embed_dim_, embed_dim_);
        Initializers::initialize_tensor(Wo_, weight_init, embed_dim_, embed_dim_);

        grad_Wq_ = Utils::Tensor({ embed_dim_, embed_dim_ }, true);
        grad_Wk_ = Utils::Tensor({ embed_dim_, embed_dim_ }, true);
        grad_Wv_ = Utils::Tensor({ embed_dim_, embed_dim_ }, true);
        grad_Wo_ = Utils::Tensor({ embed_dim_, embed_dim_ }, true);
    }

    Utils::Tensor forward(const Utils::Tensor& input) override {
        this->input_cache_ = Utils::Tensor(input.shape());
        ::cudaMemcpy(this->input_cache_.data(), input.data(), input.bytes(), ::cudaMemcpyDeviceToDevice);

        int batch_size = input.shape()[0];
        int seq_len = input.shape()[1];
        seq_len_cache_ = seq_len;

        Utils::Tensor output({ batch_size, seq_len, embed_dim_ });
        q_cache_ = Utils::Tensor({ batch_size, seq_len, embed_dim_ });
        k_cache_ = Utils::Tensor({ batch_size, seq_len, embed_dim_ });
        v_cache_ = Utils::Tensor({ batch_size, seq_len, embed_dim_ });
        attn_cache_ = Utils::Tensor({ batch_size, num_heads_, seq_len, seq_len });
        context_cache_ = Utils::Tensor({ batch_size, seq_len, embed_dim_ });

        cuda::attentions::launchMHAForward(
            static_cast<float*>(this->input_cache_.data()),
            static_cast<float*>(Wq_.data()),
            static_cast<float*>(Wk_.data()),
            static_cast<float*>(Wv_.data()),
            static_cast<float*>(Wo_.data()),
            static_cast<float*>(output.data()),
            static_cast<float*>(q_cache_.data()),
            static_cast<float*>(k_cache_.data()),
            static_cast<float*>(v_cache_.data()),
            static_cast<float*>(attn_cache_.data()),
            static_cast<float*>(context_cache_.data()),
            batch_size, seq_len, embed_dim_, num_heads_
        );

        return output;
    }

    Utils::Tensor backward(const Utils::Tensor& grad_output) override {
        int batch_size = grad_output.shape()[0];
        int seq_len = grad_output.shape()[1];
        Utils::Tensor grad_input({ batch_size, seq_len, embed_dim_ });

        cuda::attentions::launchMHABackward(
            static_cast<float*>(grad_output.data()),
            static_cast<float*>(this->input_cache_.data()),
            static_cast<float*>(Wq_.data()),
            static_cast<float*>(Wk_.data()),
            static_cast<float*>(Wv_.data()),
            static_cast<float*>(Wo_.data()),
            static_cast<float*>(q_cache_.data()),
            static_cast<float*>(k_cache_.data()),
            static_cast<float*>(v_cache_.data()),
            static_cast<float*>(attn_cache_.data()),
            static_cast<float*>(context_cache_.data()),
            static_cast<float*>(grad_input.data()),
            static_cast<float*>(grad_Wq_.data()),
            static_cast<float*>(grad_Wk_.data()),
            static_cast<float*>(grad_Wv_.data()),
            static_cast<float*>(grad_Wo_.data()),
            batch_size, seq_len, embed_dim_, num_heads_
        );

        if (this->optimizer_) {
            this->optimizer_->update(Wq_, grad_Wq_);
            this->optimizer_->update(Wk_, grad_Wk_);
            this->optimizer_->update(Wv_, grad_Wv_);
            this->optimizer_->update(Wo_, grad_Wo_);
        }

        return grad_input;
    }

    size_t get_flops(int batch_size = 1) const override {
        if (seq_len_cache_ <= 0) return 0;
        int head_dim = embed_dim_ / num_heads_;
        size_t BS = static_cast<size_t>(batch_size) * seq_len_cache_;
        size_t flops = 0;
        flops += 4ull * BS * embed_dim_ * embed_dim_;
        flops += static_cast<size_t>(batch_size) * num_heads_ * seq_len_cache_ * seq_len_cache_ * head_dim * 2ull;
        return flops;
    }

    Initialization get_initialization() const override { return initialization_type_; }
    int get_input_size() const override { return embed_dim_; }
    int get_output_size() const override { return embed_dim_; }
};

}