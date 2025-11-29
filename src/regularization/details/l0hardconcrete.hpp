#ifndef Nott_REGULARIZATION_DETAILS_L0_HARD_CONCRETE_HPP
#define Nott_REGULARIZATION_DETAILS_L0_HARD_CONCRETE_HPP

#include <torch/torch.h>

#include <cmath>

#include "common.hpp"

namespace Nott::Regularization::Details {

    struct L0HardConcreteOptions {
        double coefficient{0.0};
        double beta{2.0 / 3.0};
        double gamma{-0.1};
        double zeta{1.1};
    };

    struct L0HardConcreteDescriptor {
        L0HardConcreteOptions options{};
    };

    [[nodiscard]] inline torch::Tensor penalty(const L0HardConcreteDescriptor& descriptor,
                                               const torch::Tensor& log_alpha)
    {
        const auto& options = descriptor.options;
        if (options.coefficient == 0.0 || !log_alpha.defined()) {
            return detail::zeros_like_optional(log_alpha);
        }

        const auto ratio = -options.gamma / options.zeta;
        auto expected_gate = torch::sigmoid(log_alpha - options.beta * std::log(ratio));
        expected_gate = expected_gate * (options.zeta - options.gamma) + options.gamma;
        expected_gate = torch::clamp(expected_gate, 0.0, 1.0);
        return expected_gate.sum().mul(options.coefficient);
    }

}

#endif // Nott_REGULARIZATION_DETAILS_L0_HARD_CONCRETE_HPP