#ifndef THOT_REGULARIZATION_DETAILS_CENTERINGVARIANCE_HPP
#define THOT_REGULARIZATION_DETAILS_CENTERINGVARIANCE_HPP

#include <torch/torch.h>

namespace Thot::Regularization::Details {

    struct CenteringVarianceOptions {
        double coefficient{0.0};
        double target_std{1.0};
    };

    struct CenteringVarianceDescriptor {
        CenteringVarianceOptions options{};
    };

    [[nodiscard]] inline torch::Tensor penalty(const CenteringVarianceDescriptor& descriptor,
                                               const torch::Tensor& activations)
    {
        const auto& options = descriptor.options;
        if (options.coefficient == 0.0 || activations.numel() == 0) {
            return activations.new_zeros({});
        }

        auto mean = activations.mean();
        auto std = activations.std(false);
        auto penalty_value = mean.pow(2) + (std - options.target_std).pow(2);
        return penalty_value.mul(options.coefficient);
    }

}

#endif // THOT_REGULARIZATION_DETAILS_CENTERINGVARIANCE_HPP