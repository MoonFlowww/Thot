#ifndef Nott_LATENT_ATTENTION_HPP
#define Nott_LATENT_ATTENTION_HPP
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <torch/torch.h>

#include "../attention.hpp"
#include "kernel.hpp"

namespace Nott::Attention::Details {
    struct MultiHeadLatentAttentionOptions {
        std::int64_t embed_dim{};
        std::int64_t num_heads{1};
        std::int64_t latent_dim{128};
        double dropout{0.0};
        bool bias{true};
        bool batch_first{true};
        ::Nott::Attention::Variant variant{::Nott::Attention::Variant::Full};
    };

    class MultiHeadLatentAttentionImpl : public torch::nn::Module {
    public:
        explicit MultiHeadLatentAttentionImpl(MultiHeadLatentAttentionOptions options)
            : options_(std::move(options))
        {
            if (options_.num_heads <= 0) {
                throw std::invalid_argument(
                    "Multi-head latent attention requires a positive number of heads.");
            }
            if (options_.embed_dim <= 0) {
                throw std::invalid_argument(
                    "Multi-head latent attention requires a positive embedding dimension.");
            }
            if (options_.embed_dim % options_.num_heads != 0) {
                throw std::invalid_argument(
                    "Embedding dimension must be divisible by the number of heads.");
            }
            if (options_.latent_dim <= 0) {
                throw std::invalid_argument(
                    "Latent projection dimension must be positive for multi-head latent attention.");
            }

            const auto embed_dim = options_.embed_dim;

            q_proj_ = register_module(
                "q_proj",
                torch::nn::Linear(torch::nn::LinearOptions(embed_dim, embed_dim).bias(options_.bias)));
            k_latent_proj_ = register_module(
                "k_latent_proj",
                torch::nn::Linear(torch::nn::LinearOptions(embed_dim, options_.latent_dim).bias(options_.bias)));
            v_latent_proj_ = register_module(
                "v_latent_proj",
                torch::nn::Linear(torch::nn::LinearOptions(embed_dim, options_.latent_dim).bias(options_.bias)));
            k_up_proj_ = register_module(
                "k_up_proj",
                torch::nn::Linear(torch::nn::LinearOptions(options_.latent_dim, embed_dim).bias(options_.bias)));
            v_up_proj_ = register_module(
                "v_up_proj",
                torch::nn::Linear(torch::nn::LinearOptions(options_.latent_dim, embed_dim).bias(options_.bias)));
            out_proj_ = register_module(
                "out_proj",
                torch::nn::Linear(torch::nn::LinearOptions(embed_dim, embed_dim).bias(options_.bias)));

            kernel_ = register_module("kernel", ScaledDotProductKernel(options_.dropout, options_.variant));
            dropout_ = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(options_.dropout)));
        }

        torch::Tensor forward(const torch::Tensor& query,
                              const torch::Tensor& key,
                              const torch::Tensor& value,
                              const torch::Tensor& attn_mask = {},
                              const torch::Tensor& key_padding_mask = {})
        {
            const auto head_dim = options_.embed_dim / options_.num_heads;

            const auto make_batch_first = [&](const torch::Tensor& tensor) {
                if (options_.batch_first) {
                    return tensor;
                }
                return tensor.transpose(0, 1);
            };

            auto q = make_batch_first(query);
            auto k = make_batch_first(key);
            auto v = make_batch_first(value);

            const auto batch_size = q.size(0);
            const auto target_len = q.size(1);
            const auto source_len = k.size(1);

            auto q_projected = q_proj_->forward(q)
                                  .contiguous()
                                  .view({batch_size, target_len, options_.num_heads, head_dim})
                                  .transpose(1, 2);

            auto k_latent = k_latent_proj_->forward(k);
            auto v_latent = v_latent_proj_->forward(v);

            auto k_projected = k_up_proj_->forward(k_latent)
                                   .contiguous()
                                   .view({batch_size, source_len, options_.num_heads, head_dim})
                                   .transpose(1, 2);
            auto v_projected = v_up_proj_->forward(v_latent)
                                   .contiguous()
                                   .view({batch_size, source_len, options_.num_heads, head_dim})
                                   .transpose(1, 2);

            auto attn_output =
                kernel_->forward(q_projected, k_projected, v_projected, attn_mask, key_padding_mask);
            attn_output = attn_output.transpose(1, 2).contiguous().view({batch_size, target_len, options_.embed_dim});
            auto output = out_proj_->forward(attn_output);
            output = dropout_->forward(output);

            if (!options_.batch_first) {
                output = output.transpose(0, 1);
            }

            return output;
        }

        [[nodiscard]] const MultiHeadLatentAttentionOptions& options() const noexcept { return options_; }

    private:
        MultiHeadLatentAttentionOptions options_{};
        ScaledDotProductKernel kernel_{nullptr};
        torch::nn::Linear q_proj_{nullptr};
        torch::nn::Linear k_latent_proj_{nullptr};
        torch::nn::Linear v_latent_proj_{nullptr};
        torch::nn::Linear k_up_proj_{nullptr};
        torch::nn::Linear v_up_proj_{nullptr};
        torch::nn::Linear out_proj_{nullptr};
        torch::nn::Dropout dropout_{nullptr};
    };

    TORCH_MODULE(MultiHeadLatentAttention);
}

#endif // Nott_LATENT_ATTENTION_HPP