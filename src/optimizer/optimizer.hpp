#ifndef THOT_OPTIMIZER_HPP
#define THOT_OPTIMIZER_HPP
#include <variant>
#include "details/adam.hpp"
#include "details/sgd.hpp"
#include "details/sophia.hpp"
#include "registry.hpp"
#include <functional>
#include <type_traits>

namespace Thot::Optimizer {
    using SGDOptions = Details::SGDOptions;
    using SGDDescriptor = Details::SGDDescriptor;
    using AdamWOptions = Details::AdamWOptions;
    using AdamWDescriptor = Details::AdamWDescriptor;

    using SophiaGOptions = Details::SophiaGOptions;
    using SophiaGDescriptor = Details::SophiaGDescriptor;
    using SophiaHOptions = Details::SophiaHOptions;
    using SophiaHDescriptor = Details::SophiaHDescriptor;

    using Descriptor = std::variant<SGDDescriptor, AdamWDescriptor, SophiaGDescriptor, SophiaHDescriptor>;



    [[nodiscard]] constexpr auto SGD(const SGDOptions& options = {}) noexcept -> SGDDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto AdamW(const AdamWOptions& options = {}) noexcept -> AdamWDescriptor {
        return {options};
    }
    [[nodiscard]] constexpr auto SophiaG(const SophiaGOptions& options = {}) noexcept -> SophiaGDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto SophiaH(const SophiaHOptions& options = {}) noexcept -> SophiaHDescriptor {
        return {options};
    }

}

#endif //THOT_OPTIMIZER_HPP