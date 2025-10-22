#ifndef THOT_OPTIMIZER_HPP
#define THOT_OPTIMIZER_HPP
// This file is a factory, must exempt it from any logical-code. For functions look into "/details"
#include <variant>
#include "details/adam.hpp"
#include "details/sgd.hpp"
#include "registry.hpp"

namespace Thot::Optimizer {
    using SGDOptions = Details::SGDOptions;
    using SGDDescriptor = Details::SGDDescriptor;
    using AdamWOptions = Details::AdamWOptions;
    using AdamWDescriptor = Details::AdamWDescriptor;

    using Descriptor = std::variant<SGDDescriptor, AdamWDescriptor>;

    [[nodiscard]] constexpr auto SGD(const SGDOptions& options = {}) noexcept -> SGDDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto AdamW(const AdamWOptions& options = {}) noexcept -> AdamWDescriptor {
        return {options};
    }
}

#endif //THOT_OPTIMIZER_HP