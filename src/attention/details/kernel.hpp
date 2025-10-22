#ifndef THOT_KERNEL_HPP
#define THOT_KERNEL_HPP
#include <cmath>
#include <limits>

#include <torch/torch.h>

#include "../attention.hpp"

namespace Thot::Attention::Details {
    class ScaledDotProductKernelImpl : public torch::nn::Module {
    public:
        explicit ScaledDotProductKernelImpl(double dropout = 0.0,
                                            ::Thot::Attention::Variant variant = ::Thot::Attention::Variant::Full)
            : variant_(variant),
              dropout_(register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(dropout)))) {}

        torch::Tensor forward(const torch::Tensor& query,
                              const torch::Tensor& key,
                              const torch::Tensor& value,
                              const torch::Tensor& attn_mask = {},
                              const torch::Tensor& key_padding_mask = {})
        {
            auto scores = torch::matmul(query, key.transpose(-2, -1));
            const auto head_dim = query.size(-1);
            if (head_dim > 0) {
                scores = scores / std::sqrt(static_cast<double>(head_dim));
            }

            if (key_padding_mask.defined() && key_padding_mask.numel() > 0) {
                auto mask = key_padding_mask.to(torch::kBool).unsqueeze(1).unsqueeze(2);
                scores = scores.masked_fill(mask, -std::numeric_limits<float>::infinity());
            }

            if (attn_mask.defined() && attn_mask.numel() > 0) {
                if (attn_mask.dim() == 2) {
                    scores = scores + attn_mask.unsqueeze(0).unsqueeze(0);
                } else if (attn_mask.dim() == 3) {
                    scores = scores + attn_mask.unsqueeze(0);
                } else {
                    scores = scores + attn_mask;
                }
            }

            if (variant_ == ::Thot::Attention::Variant::Causal) {
                const auto seq_len = scores.size(-1);
                const auto tgt_len = scores.size(-2);
                auto causal_mask = torch::ones({tgt_len, seq_len}, scores.options().dtype(torch::kBool)).triu(1);
                scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -std::numeric_limits<float>::infinity());
            }

            auto attn = torch::softmax(scores, -1);
            attn = dropout_->forward(attn);
            return torch::matmul(attn, value);
        }

    private:
        ::Thot::Attention::Variant variant_{::Thot::Attention::Variant::Full};
        torch::nn::Dropout dropout_{nullptr};
    };

    TORCH_MODULE(ScaledDotProductKernel);
}
#endif //THOT_KERNEL_HPP