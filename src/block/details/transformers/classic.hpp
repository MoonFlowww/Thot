#ifndef THOT_CLASSIC_HPP
#define THOT_CLASSIC_HPP


#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <utility>
#include <vector>

#include "../../../activation/activation.hpp"
#include "../../../attention/attention.hpp"
#include "../../../initialization/initialization.hpp"
#include "../../../layer/layer.hpp"
#include "../residual.hpp"
#include "../sequential.hpp"

namespace Thot::Block::Details::Transformer::Classic {
    enum class PositionalEncodingType {
        None,
        Sinusoidal,
        Learned,
    };

    struct AttentionOptions {
        std::int64_t embed_dim{512};
        std::int64_t num_heads{8};
        double dropout{0.0};
        bool bias{true};
        bool batch_first{true};
        ::Thot::Attention::Variant variant{::Thot::Attention::Variant::Full};
    };

    struct FeedForwardOptions {
        std::int64_t embed_dim{512};
        double mlp_ratio{4.0};
        ::Thot::Activation::Descriptor activation{::Thot::Activation::GeLU};
        bool bias{true};
        ::Thot::Initialization::Descriptor initialization{::Thot::Initialization::Default};
    };

    struct LayerNormOptions {
        double eps{1e-5};
        bool elementwise_affine{true};
    };

    struct PositionalEncodingOptions {
        PositionalEncodingType type{PositionalEncodingType::None};
        double dropout{0.0};
        std::size_t max_length{2048};
    };

    struct EncoderLayerDescriptor {
        ::Thot::Attention::Descriptor attention;
        ::Thot::Layer::Descriptor attention_dropout;
        std::vector<::Thot::Layer::Descriptor> feed_forward;
        ::Thot::Layer::Descriptor feed_forward_dropout;
    };

    struct EncoderOptions {
        std::size_t layers{1};
        std::int64_t embed_dim{512};
        AttentionOptions attention{};
        FeedForwardOptions feed_forward{};
        LayerNormOptions layer_norm{};
        PositionalEncodingOptions positional_encoding{};
        double dropout{0.0};
    };

    struct EncoderDescriptor {
        EncoderOptions options{};
        std::vector<EncoderLayerDescriptor> layers{};
    };

    [[nodiscard]] inline auto Encoder(const EncoderOptions& options) -> EncoderDescriptor
    {
        EncoderDescriptor descriptor{};
        descriptor.options = options;

        auto attention_options = options.attention;
        attention_options.embed_dim = options.embed_dim;

        auto feed_forward = options.feed_forward;
        feed_forward.embed_dim = options.embed_dim;

        auto hidden_dim = static_cast<std::int64_t>(std::llround(feed_forward.mlp_ratio * static_cast<double>(options.embed_dim)));
        if (hidden_dim <= 0) {
            hidden_dim = options.embed_dim;
        }

        for (std::size_t index = 0; index < options.layers; ++index) {
            EncoderLayerDescriptor layer{};

            ::Thot::Attention::MultiHeadOptions attention_descriptor_options{};
            attention_descriptor_options.embed_dim = attention_options.embed_dim;
            attention_descriptor_options.num_heads = attention_options.num_heads;
            attention_descriptor_options.dropout = attention_options.dropout;
            attention_descriptor_options.bias = attention_options.bias;
            attention_descriptor_options.batch_first = attention_options.batch_first;
            attention_descriptor_options.variant = attention_options.variant;
            layer.attention = ::Thot::Attention::MultiHead(attention_descriptor_options);

            layer.attention_dropout = ::Thot::Layer::Dropout({attention_options.dropout});

            ::Thot::Layer::FCOptions fc1_options{feed_forward.embed_dim, hidden_dim, feed_forward.bias};
            layer.feed_forward.emplace_back(::Thot::Layer::FC(fc1_options, feed_forward.activation, feed_forward.initialization));

            ::Thot::Layer::FCOptions fc2_options{hidden_dim, feed_forward.embed_dim, feed_forward.bias};
            layer.feed_forward.emplace_back(::Thot::Layer::FC(fc2_options, ::Thot::Activation::Identity, feed_forward.initialization));

            layer.feed_forward_dropout = ::Thot::Layer::Dropout({options.dropout});

            descriptor.layers.emplace_back(std::move(layer));
        }

        return descriptor;
    }

    struct DecoderLayerDescriptor {
        ::Thot::Attention::Descriptor self_attention;
        ::Thot::Layer::Descriptor self_attention_dropout;
        ::Thot::Attention::Descriptor cross_attention;
        ::Thot::Layer::Descriptor cross_attention_dropout;
        std::vector<::Thot::Layer::Descriptor> feed_forward;
        ::Thot::Layer::Descriptor feed_forward_dropout;
    };

    struct DecoderOptions {
        std::size_t layers{1};
        std::int64_t embed_dim{512};
        AttentionOptions self_attention{};
        AttentionOptions cross_attention{};
        FeedForwardOptions feed_forward{};
        LayerNormOptions layer_norm{};
        PositionalEncodingOptions positional_encoding{};
        double dropout{0.0};
    };

    struct DecoderDescriptor {
        DecoderOptions options{};
        std::vector<DecoderLayerDescriptor> layers{};
    };

    [[nodiscard]] inline auto Decoder(const DecoderOptions& options) -> DecoderDescriptor {
        DecoderDescriptor descriptor{};
        descriptor.options = options;

        auto self_attention = options.self_attention;
        self_attention.embed_dim = options.embed_dim;

        auto cross_attention = options.cross_attention;
        cross_attention.embed_dim = options.embed_dim;

        auto feed_forward = options.feed_forward;
        feed_forward.embed_dim = options.embed_dim;

        auto hidden_dim = static_cast<std::int64_t>(std::llround(feed_forward.mlp_ratio * static_cast<double>(options.embed_dim)));
        if (hidden_dim <= 0) {
            hidden_dim = options.embed_dim;
        }

        for (std::size_t index = 0; index < options.layers; ++index) {
            DecoderLayerDescriptor layer{};

            ::Thot::Attention::MultiHeadOptions self_attention_descriptor_options{};
            self_attention_descriptor_options.embed_dim = self_attention.embed_dim;
            self_attention_descriptor_options.num_heads = self_attention.num_heads;
            self_attention_descriptor_options.dropout = self_attention.dropout;
            self_attention_descriptor_options.bias = self_attention.bias;
            self_attention_descriptor_options.batch_first = self_attention.batch_first;
            self_attention_descriptor_options.variant = self_attention.variant;
            layer.self_attention = ::Thot::Attention::MultiHead(self_attention_descriptor_options);
            layer.self_attention_dropout = ::Thot::Layer::Dropout({self_attention.dropout});

            ::Thot::Attention::MultiHeadOptions cross_attention_descriptor_options{};
            cross_attention_descriptor_options.embed_dim = cross_attention.embed_dim;
            cross_attention_descriptor_options.num_heads = cross_attention.num_heads;
            cross_attention_descriptor_options.dropout = cross_attention.dropout;
            cross_attention_descriptor_options.bias = cross_attention.bias;
            cross_attention_descriptor_options.batch_first = cross_attention.batch_first;
            cross_attention_descriptor_options.variant = cross_attention.variant;
            layer.cross_attention = ::Thot::Attention::MultiHead(cross_attention_descriptor_options);
            layer.cross_attention_dropout = ::Thot::Layer::Dropout({cross_attention.dropout});

            ::Thot::Layer::FCOptions fc1_options{feed_forward.embed_dim, hidden_dim, feed_forward.bias};
            layer.feed_forward.emplace_back(::Thot::Layer::FC(fc1_options, feed_forward.activation, feed_forward.initialization));

            ::Thot::Layer::FCOptions fc2_options{hidden_dim, feed_forward.embed_dim, feed_forward.bias};
            layer.feed_forward.emplace_back(::Thot::Layer::FC(fc2_options, ::Thot::Activation::Identity, feed_forward.initialization));

            layer.feed_forward_dropout = ::Thot::Layer::Dropout({options.dropout});

            descriptor.layers.emplace_back(std::move(layer));
        }

        return descriptor;
    }
}
#endif //THOT_CLASSIC_HPP