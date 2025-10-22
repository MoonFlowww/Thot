#ifndef THOT_LAYER_HPP
#define THOT_LAYER_HPP
// This file is a factory, must exempt it from any logical-code. For functions look into "/details"

#include "layer/details/fc.hpp"

namespace Thot::Layer {
    using FCOptions = Details::FCOptions;
    using FCDescriptor = Details::FCDescriptor;

    [[nodiscard]] constexpr auto FC(const FCOptions& options,
                                    ::Thot::Activation::Descriptor activation = ::Thot::Activation::Identity,
                                    ::Thot::Initialization::Descriptor initialization = ::Thot::Initialization::Default)
        -> FCDescriptor {
        return {options, activation, initialization};
    }
}

#endif //THOT_LAYER_HPP
