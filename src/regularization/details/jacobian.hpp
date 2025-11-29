#ifndef OMNI_REGULARIZATION_DETAILS_JACOBIANNORM_HPP
#define OMNI_REGULARIZATION_DETAILS_JACOBIANNORM_HPP

#include <torch/torch.h>

namespace Omni::Regularization::Details {

    struct JacobianNormOptions {
        double coefficient{0.0};
    };

    struct JacobianNormDescriptor {
        JacobianNormOptions options{};
    };

    [[nodiscard]] inline torch::Tensor penalty(const JacobianNormDescriptor& descriptor, const torch::Tensor& jacobian)
    {
        const auto& options = descriptor.options;
        if (options.coefficient == 0.0 || jacobian.numel() == 0) {
            return jacobian.new_zeros({});
        }

        return jacobian.pow(2).sum().mul(options.coefficient);
    }

}

#endif // OMNI_REGULARIZATION_DETAILS_JACOBIANNORM_HPP