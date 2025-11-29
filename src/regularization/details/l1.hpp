#ifndef OMNI_REGULARIZATION_DETAILS_L1_HPP
#define OMNI_REGULARIZATION_DETAILS_L1_HPP

#include <torch/torch.h>

namespace Omni::Regularization::Details {

    struct L1Options {
        double coefficient{0.0};
    };

    struct L1Descriptor {
        L1Options options{};
    };

    [[nodiscard]] inline torch::Tensor penalty(const L1Descriptor& descriptor, const torch::Tensor& params)
    {
        const auto coefficient = descriptor.options.coefficient;
        if (coefficient == 0.0) {
            return params.new_zeros({});
        }

        return params.abs().sum().mul(coefficient);
    }

}

#endif // OMNI_REGULARIZATION_DETAILS_L1_HPP