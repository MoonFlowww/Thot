#ifndef THOT_REGULARIZATION_HPP
#define THOT_REGULARIZATION_HPP
// This file is an factory, must exempt it from any logical-code. For functions look into "/details"
#include <variant>

#include "details/ewc.hpp"
#include "details/l2.hpp"
#include "details/mas.hpp"
#include "details/nuclear_norm.hpp"
#include "details/si.hpp"

namespace Thot::Regularization {

    using L2Options = Details::L2Options;
    using L2Descriptor = Details::L2Descriptor;

    using EWCOptions = Details::EWCOptions;
    using EWCDescriptor = Details::EWCDescriptor;

    using MASOptions = Details::MASOptions;
    using MASDescriptor = Details::MASDescriptor;

    using SIOptions = Details::SIOptions;
    using SIDescriptor = Details::SIDescriptor;

    using NuclearNormOptions = Details::NuclearNormOptions;
    using NuclearNormDescriptor = Details::NuclearNormDescriptor;

    using Descriptor = std::variant<
        L2Descriptor,
        EWCDescriptor,
        MASDescriptor,
        SIDescriptor,
        NuclearNormDescriptor
    >;

    [[nodiscard]] constexpr auto L2(const L2Options& options = {}) noexcept -> L2Descriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto EWC(const EWCOptions& options = {}) noexcept -> EWCDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto MAS(const MASOptions& options = {}) noexcept -> MASDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto SI(const SIOptions& options = {}) noexcept -> SIDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto NuclearNorm(const NuclearNormOptions& options = {}) noexcept
        -> NuclearNormDescriptor {
        return {options};
    }

}

#endif //THOT_REGULARIZATION_HPP