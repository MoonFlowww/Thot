#ifndef THOT_FLATTEN_HPP
#define THOT_FLATTEN_HPP
#include <cstdint>

#include "../../activation/activation.hpp"

namespace Thot::Layer::Details {

    struct FlattenOptions {
        std::int64_t start_dim{1};
        std::int64_t end_dim{-1};
    };

    struct FlattenDescriptor {
        FlattenOptions options{};
        ::Thot::Activation::Descriptor activation{::Thot::Activation::Identity};
    };

}

#endif //THOT_FLATTEN_HPP