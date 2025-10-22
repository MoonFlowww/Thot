#ifndef THOT_BLOCK_DETAILS_RESIDUAL_HPP
#define THOT_BLOCK_DETAILS_RESIDUAL_HPP

#include <cstddef>
#include <optional>
#include <vector>

#include "../../activation/activation.hpp"
#include "../../layer/layer.hpp"

namespace Thot::Block::Details {
    struct ResidualSkipOptions {
        bool use_projection{false};
        std::optional<::Thot::Layer::Descriptor> projection{};
    };

    struct ResidualOutputOptions {
        ::Thot::Activation::Descriptor final_activation{::Thot::Activation::Identity};
        double dropout{0.0};
    };

    struct ResidualDescriptor {
        std::vector<::Thot::Layer::Descriptor> layers{};
        std::size_t repeats{1};
        ResidualSkipOptions skip{};
        ResidualOutputOptions output{};
    };
}

#endif // THOT_BLOCK_DETAILS_RESIDUAL_HPP