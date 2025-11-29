#ifndef Nott_REGULARIZATION_DETAILS_ELASTICNET_HPP
#define Nott_REGULARIZATION_DETAILS_ELASTICNET_HPP

#include <torch/torch.h>

namespace Nott::Regularization::Details {

    struct ElasticNetOptions {
        double l1_coefficient{0.0};
        double l2_coefficient{0.0};
    };

    struct ElasticNetDescriptor {
        ElasticNetOptions options{};
    };

    [[nodiscard]] inline torch::Tensor penalty(const ElasticNetDescriptor& descriptor, const torch::Tensor& params)
    {
        const auto& options = descriptor.options;
        if (options.l1_coefficient == 0.0 && options.l2_coefficient == 0.0) {
            return params.new_zeros({});
        }

        auto penalty_value = params.new_zeros({});
        if (options.l1_coefficient != 0.0) {
            penalty_value = penalty_value + params.abs().sum().mul(options.l1_coefficient);
        }
        if (options.l2_coefficient != 0.0) {
            penalty_value = penalty_value + params.pow(2).sum().mul(options.l2_coefficient);
        }
        return penalty_value;
    }

}

#endif // Nott_REGULARIZATION_DETAILS_ELASTICNET_HPP