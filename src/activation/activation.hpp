#ifndef THOT_ACTIVATION_HPP
#define THOT_ACTIVATION_HPP
// This file is an factory, must exempt it from any logical-code. For functions look into "/details"

#include "activation/details/relu.hpp"

namespace Thot::Activation {
    [[nodiscard]] constexpr auto make_relu() -> ::Thot::Activation::Details::ReLU {
        return {};
    }
}

#endif //THOT_ACTIVATION_HPP