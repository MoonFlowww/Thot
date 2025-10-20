#ifndef THOT_OPTIMIZER_HPP
#define THOT_OPTIMIZER_HPP
// This file is an factory, must exempt it from any logical-code. For functions look into "/details"
#include "optimizer/details/sgd.hpp"

namespace thot::optimizer {
    template <typename... Options>
    [[nodiscard]] constexpr auto make_sgd() noexcept {
        return details::sgd_factory<Options...>{};
    }
}
#endif //THOT_OPTIMIZER_HPP