#ifndef THOT_OPTIMIZER_HPP
#define THOT_OPTIMIZER_HPP
// This file is a factory, must exempt it from any logical-code. For functions look into "/details"
#include <variant>
#include "details/sgd.hpp"
#include "registry.hpp"

namespace Thot::Optimizer {
    using SGDOptions = Details::SGDOptions;
    using SGDDescriptor = Details::SGDDescriptor;

    using Descriptor = std::variant<SGDDescriptor>;

    [[nodiscard]] constexpr auto SGD(const SGDOptions& options = {}) noexcept -> SGDDescriptor {
        return {options};
    }
}

#endif //THOT_OPTIMIZER_HP