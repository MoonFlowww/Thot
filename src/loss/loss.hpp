#ifndef THOT_LOSS_HPP
#define THOT_LOSS_HPP
// This file is a factory, must exempt it from any logical-code. For functions look into "/details"

#include "details/reduction.hpp"
#include "details/ce.hpp"
#include "details/mse.hpp"

namespace Thot::Loss {
    using Reduction = Details::Reduction;
    using MSEOptions = Details::MSEOptions;
    using MSEDescriptor = Details::MSEDescriptor;
    using CrossEntropyOptions = Details::CrossEntropyOptions;
    using CrossEntropyDescriptor = Details::CrossEntropyDescriptor;

    [[nodiscard]] constexpr auto MSE(const MSEOptions& options = {}) noexcept -> MSEDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto CrossEntropy(const CrossEntropyOptions& options = {}) noexcept -> CrossEntropyDescriptor {
        return {options};
    }
}

#endif //THOT_LOSS_HPP
