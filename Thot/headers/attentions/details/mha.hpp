#pragma once

#include "../../tensor.hpp"
#include "../../initializations/initializations.hpp"
#include "../../optimizations/optimizations.hpp"
#include "../../../cuda/cuh/attentions/mha.cuh"

namespace Thot {

    class MHAAtt : public Attention {
    private:
        int embed_dim_;
        int num_heads_;
        int head_dim_;
        Initialization initialization_type_;
        Utils::Tensor Wq_, Wk_, Wv_, Wo_;
        Utils::Tensor grad_Wq_, grad_Wk_, grad_Wv_, grad_Wo_;
        Utils::Tensor Q_, K_, V_, softmax_, context_;

    public:
        MHAAtt(int embed_dim, int num_heads,
               Initialization init = Initialization::Xavier,
               const std::string& name = "MHAAtt")
            : Attention(name), embed_dim_(embed_dim), num_heads_(num_heads),
              head_dim_(embed_dim / num_heads), initialization_type_(init) {
            Wq_ = Utils::Tensor({embed_dim_, embed_dim_});
            Wk_ = Utils::Tensor({embed_dim_, embed_dim_});
            Wv_ = Utils::Tensor({embed_dim_, embed_dim_});
            Wo_ = Utils::Tensor({embed_dim_, embed_dim_});
            Initializers::initialize_tensor(Wq_, init, embed_dim_, embed_dim_);
            Initializers::initialize_tensor(Wk_, init, embed_dim_, embed_dim_);
            Initializers::initialize_tensor(Wv_, init, embed_dim_, embed_dim_);
            Initializers::initialize_tensor(Wo_, init, embed_dim_, embed_dim_);
            grad_Wq_ = Utils::Tensor({embed_dim_, embed_dim_}, true);
            grad_Wk_ = Utils::Tensor({embed_dim_, embed_dim_}, true);
            grad_Wv_ = Utils::Tensor({embed_dim_, embed_dim_}, true);
            grad_Wo_ = Utils::Tensor({embed_dim_, embed_dim_}, true);
        }

        Utils::Tensor forward(const Utils::Tensor& input) override {
            this->input_cache_ = Utils::Tensor(input.shape());
            ::cudaMemcpy(this->input_cache_.data(), input.data(), input.bytes(), cudaMemcpyDeviceToDevice);
            int batch_size = input.shape()[0];
            int seq_len = input.shape()[1];
            Q_ = Utils::Tensor({batch_size, num_heads_, seq_len, head_dim_});
            K_ = Utils::Tensor({batch_size, num_heads_, seq_len, head_dim_});
            V_ = Utils::Tensor({batch_size, num_heads_, seq_len, head_dim_});
            softmax_ = Utils::Tensor({batch_size, num_heads_, seq_len, seq_len});
            context_ = Utils::Tensor({batch_size, seq_len, embed_dim_});
            Utils::Tensor output({batch_size, seq_len, embed_dim_});
            ::cuda::attentions::launchMHAForward(
                static_cast<const float*>(this->input_cache_.data()),
                static_cast<const float*>(Wq_.data()),
                static_cast<const float*>(Wk_.data()),
                static_cast<const float*>(Wv_.data()),
                static_cast<const float*>(Wo_.data()),
                static_cast<float*>(Q_.data()),
                static_cast<float*>(K_.data()),
                static_cast<float*>(V_.data()),
                static_cast<float*>(softmax_.data()),
                static_cast<float*>(context_.data()),
                static_cast<float*>(output.data()),
                batch_size, seq_len, embed_dim_, num_heads_);
            return std::move(output);
        }

        Utils::Tensor backward(const Utils::Tensor& grad_output) override {
            int batch_size = grad_output.shape()[0];
            int seq_len = grad_output.shape()[1];
            Utils::Tensor grad_input({batch_size, seq_len, embed_dim_});
            ::cuda::attentions::launchMHABackward(
                static_cast<const float*>(grad_output.data()),
                static_cast<float*>(grad_input.data()),
                static_cast<float*>(grad_Wq_.data()),
                static_cast<float*>(grad_Wk_.data()),
                static_cast<float*>(grad_Wv_.data()),
                static_cast<float*>(grad_Wo_.data()),
                batch_size * seq_len, embed_dim_);
            if (optimizer_) {
                optimizer_->update(Wq_, grad_Wq_);
                optimizer_->update(Wk_, grad_Wk_);
                optimizer_->update(Wv_, grad_Wv_);
                optimizer_->update(Wo_, grad_Wo_);
            }
            return std::move(grad_input);
        }

        size_t get_flops(int batch_size = 1) const override { return 0; }
        Activation get_activation() const override { return Activation::Linear; }
        Initialization get_initialization() const override { return initialization_type_; }
        int get_input_size() const override { return embed_dim_; }
        int get_output_size() const override { return embed_dim_; }
    };

} // namespace Thot