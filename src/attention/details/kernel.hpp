#ifndef THOT_KERNEL_HPP
#define THOT_KERNEL_HPP
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

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
            const auto batch_size = scores.size(0);
            const auto num_heads = scores.size(1);
            const auto target_len = scores.size(2);
            const auto source_len = scores.size(3);
            if (head_dim > 0) {
                scores = scores / std::sqrt(static_cast<double>(head_dim));
            }

            if (key_padding_mask.defined() && key_padding_mask.numel() > 0) {
                auto mask = key_padding_mask.to(torch::kBool).unsqueeze(1).unsqueeze(2);
                scores = scores.masked_fill(mask, -std::numeric_limits<float>::infinity());
            }

            if (attn_mask.defined() && attn_mask.numel() > 0) {
auto mask = attn_mask;

                const auto make_size_mismatch_error = [&](const std::string& reason) {
                    throw std::invalid_argument("Attention mask dimensions mismatch: " + reason +
                                                ". Expected (batch=" + std::to_string(batch_size) +
                                                ", heads=" + std::to_string(num_heads) +
                                                ", target=" + std::to_string(target_len) +
                                                ", source=" + std::to_string(source_len) + ") but got " +
                                                std::to_string(mask.dim()) + "D mask.");
                };

                switch (mask.dim()) {
                case 2:
                    if (mask.size(0) != target_len || mask.size(1) != source_len) {
                        make_size_mismatch_error("2D mask must match target and source dimensions");
                    }
                    mask = mask.unsqueeze(0).unsqueeze(0);
                    break;
                case 3:
                    if (mask.size(1) != target_len || mask.size(2) != source_len) {
                        make_size_mismatch_error("3D mask must match target and source dimensions in the last two axes");
                    }

                    if (mask.size(0) == batch_size) {
                        mask = mask.unsqueeze(1);
                    } else if (mask.size(0) == batch_size * num_heads) {
                        mask = mask.view({batch_size, num_heads, target_len, source_len});
                    } else {
                        make_size_mismatch_error("3D mask batch dimension must equal batch_size or batch_size * num_heads");
                    }
                    break;
                case 4:
                    if (mask.size(0) != batch_size || mask.size(1) != num_heads || mask.size(2) != target_len ||
                        mask.size(3) != source_len) {
                        make_size_mismatch_error("4D mask must match (batch, heads, target, source)");
                    }
                    break;
                default:
                    throw std::invalid_argument("Unsupported attention mask dimensionality: " + std::to_string(mask.dim()));
                }
                mask = mask.to(scores.dtype());
                scores = scores + mask;
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