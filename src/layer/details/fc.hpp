#ifndef THOT_FC_HPP
#define THOT_FC_HPP

#include <cstdint>

#include "activation/activation.hpp"
#include "initialization/initialization.hpp"

namespace Thot::Layer::Details {

struct FCOptions {
    std::int64_t in_features{};
    std::int64_t out_features{};
    bool bias{true};
};

struct FCDescriptor {
    FCOptions options;
    ::Thot::Activation::Descriptor activation{::Thot::Activation::Identity};
    ::Thot::Initialization::Descriptor initialization{::Thot::Initialization::Default};
};

}  // namespace Thot::Layer::Details

#endif //THOT_FC_HPP
