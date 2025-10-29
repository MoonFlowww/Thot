#ifndef THOT_REGULARIZATION_DETAILS_R1_HPP
#define THOT_REGULARIZATION_DETAILS_R1_HPP

#include <torch/torch.h>

#include "common.hpp"

namespace Thot::Regularization::Details {

    struct R1Options {
        double coefficient{0.0};
    };

    struct R1Descriptor {
        R1Options options{};
    };

    [[nodiscard]] inline torch::Tensor penalty(const R1Descriptor& descriptor, const torch::Tensor& gradients)
    {
        const auto& options = descriptor.options;
        if (options.coefficient == 0.0 || gradients.numel() == 0 || gradients.dim() == 0) {
            return detail::zeros_like_optional(gradients);
        }

        auto tensor = gradients.view({gradients.size(0), -1});
        auto penalty_value = tensor.pow(2).sum(1).mean();
        return penalty_value.mul(options.coefficient);
    }

}

#endif // THOT_REGULARIZATION_DETAILS_R1_HPP