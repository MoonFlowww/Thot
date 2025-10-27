#ifndef THOT_ATTENTION_HPP
#define THOT_ATTENTION_HPP
// This file is an factory, must exempt it from any logical-code. For functions look into "/details"

#include <cstdint>
#include <variant>

namespace Thot::Attention {
    enum class Variant {
        Full,
        Causal,
    };

    struct MultiHeadOptions {
        std::int64_t embed_dim{};
        std::int64_t num_heads{1};
        double dropout{0.0};
        bool bias{true};
        bool add_bias_kv{false};
        bool add_zero_attn{false};
        bool batch_first{true};
        Variant variant{Variant::Full};
    };

    struct MultiHeadDescriptor {
        MultiHeadOptions options{};
    };

    using Descriptor = std::variant<MultiHeadDescriptor>;

    [[nodiscard]] inline auto MultiHead(const MultiHeadOptions& options) -> Descriptor
    {
        return Descriptor{MultiHeadDescriptor{options}};
    }
}

#endif //THOT_ATTENTION_HPP