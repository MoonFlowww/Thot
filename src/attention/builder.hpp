#ifndef OMNI_ATTENTION_BUILDER_HPP
#define OMNI_ATTENTION_BUILDER_HPP

#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include <torch/torch.h>

#include "attention.hpp"
#include "details/head.hpp"
#include "details/latent.hpp"

namespace Omni::Attention::Details {
    using AttentionModule = std::variant<MultiHeadAttention, MultiHeadLatentAttention>;

    template <typename Module>
    AttentionModule register_attention(Module& module,
                                       const std::string& name,
                                       ::Omni::Attention::Descriptor descriptor)
    {
        return std::visit(
            [&](auto&& attention_descriptor) -> AttentionModule {
                using Descriptor = std::decay_t<decltype(attention_descriptor)>;
                if constexpr (std::is_same_v<Descriptor, ::Omni::Attention::MultiHeadDescriptor>) {
                    ::Omni::Attention::Details::MultiHeadAttentionOptions attention_options{};
                    attention_options.embed_dim = attention_descriptor.options.embed_dim;
                    attention_options.num_heads = attention_descriptor.options.num_heads;
                    attention_options.dropout = attention_descriptor.options.dropout;
                    attention_options.bias = attention_descriptor.options.bias;
                    attention_options.batch_first = attention_descriptor.options.batch_first;
                    attention_options.variant = attention_descriptor.options.variant;
                    return module.register_module(name,
                                                  ::Omni::Attention::Details::MultiHeadAttention(attention_options));
                } else if constexpr (std::is_same_v<Descriptor, ::Omni::Attention::MultiHeadLatentDescriptor>) {
                    ::Omni::Attention::Details::MultiHeadLatentAttentionOptions attention_options{};
                    attention_options.embed_dim = attention_descriptor.options.embed_dim;
                    attention_options.num_heads = attention_descriptor.options.num_heads;
                    attention_options.latent_dim = attention_descriptor.options.latent_dim;
                    attention_options.dropout = attention_descriptor.options.dropout;
                    attention_options.bias = attention_descriptor.options.bias;
                    attention_options.batch_first = attention_descriptor.options.batch_first;
                    attention_options.variant = attention_descriptor.options.variant;
                    return module.register_module(
                        name, ::Omni::Attention::Details::MultiHeadLatentAttention(attention_options));
                } else {
                    throw std::invalid_argument("Unsupported attention descriptor provided.");
                }
            },
            std::move(descriptor));
    }

    inline torch::Tensor forward_attention(AttentionModule& module,
                                           const torch::Tensor& query,
                                           const torch::Tensor& key,
                                           const torch::Tensor& value,
                                           const torch::Tensor& attn_mask,
                                           const torch::Tensor& key_padding_mask)
    {
        return std::visit(
            [&](auto& attention) {
                return attention->forward(query, key, value, attn_mask, key_padding_mask);
            },
            module);
    }
}

#endif // OMNI_ATTENTION_BUILDER_HPP