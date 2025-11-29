#ifndef OMNI_REGULARIZATION_DETAILS_R2_HPP
#define OMNI_REGULARIZATION_DETAILS_R2_HPP

#include <torch/torch.h>

#include "common.hpp"

namespace Omni::Regularization::Details {

    struct R2Options {
        double coefficient{0.0};
    };

    struct R2Descriptor {
        R2Options options{};
    };

    [[nodiscard]] inline torch::Tensor penalty(const R2Descriptor& descriptor, const torch::Tensor& gradients)
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

#endif // OMNI_REGULARIZATION_DETAILS_R2_HPP