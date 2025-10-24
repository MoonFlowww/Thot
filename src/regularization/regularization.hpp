#ifndef THOT_REGULARIZATION_HPP
#define THOT_REGULARIZATION_HPP
// This file is an factory, must exempt it from any logical-code. For functions look into "/details"
#include <variant>

#include "details/ewc.hpp"
#include "details/l2.hpp"
#include "details/mas.hpp"
#include "details/nuclearnorm.hpp"
#include "details/si.hpp"
#include "details/fge.hpp"
#include "details/sfge.hpp"
#include "details/swa.hpp"
#include "details/swag.hpp"

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

    using SWAOptions = Details::SWAOptions;
    using SWADescriptor = Details::SWADescriptor;

    using SWAGOptions = Details::SWAGOptions;
    using SWAGDescriptor = Details::SWAGDescriptor;

    using FGEOptions = Details::FGEOptions;
    using FGEDescriptor = Details::FGEDescriptor;

    using SFGEOptions = Details::SFGEOptions;
    using SFGEDescriptor = Details::SFGEDescriptor;

    using Descriptor = std::variant<
        L2Descriptor,
        EWCDescriptor,
        MASDescriptor,
        SIDescriptor,
        NuclearNormDescriptor,
        SWADescriptor,
        SWAGDescriptor,
        FGEDescriptor,
        SFGEDescriptor>;

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

    [[nodiscard]] constexpr auto SWA(const SWAOptions& options = {}) noexcept -> SWADescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto SWAG(const SWAGOptions& options = {}) noexcept -> SWAGDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto FGE(const FGEOptions& options = {}) noexcept -> FGEDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto SFGE(const SFGEOptions& options = {}) noexcept -> SFGEDescriptor {
        return {options};
    }

}

#endif //THOT_REGULARIZATION_HPP