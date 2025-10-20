#ifndef THOT_LAYER_HPP
#define THOT_LAYER_HPP
// This file is an factory, must exempt it from any logical-code. For functions look into "/details"

#include <cstddef>

#include "details/fc.hpp"

namespace thot::layer {
    template <std::size_t In, std::size_t Out, class Init = initialization::Xavier>
    using linear = details::linear_descriptor<In, Out, Init>;

    template <std::size_t In, std::size_t Out, class Init = initialization::Xavier>
    inline auto make_linear() {
        return linear<In, Out, Init>{};
    }
}
#endif //THOT_LAYER_HPP