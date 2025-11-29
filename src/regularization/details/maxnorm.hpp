#ifndef OMNI_REGULARIZATION_DETAILS_MAXNORM_HPP
#define OMNI_REGULARIZATION_DETAILS_MAXNORM_HPP

#include <torch/torch.h>

#include "common.hpp"

namespace Omni::Regularization::Details {

    struct MaxNormOptions {
        double coefficient{0.0};
        double max_norm{1.0};
        std::int64_t dim{0};
    };

    struct MaxNormDescriptor {
        MaxNormOptions options{};
    };

    [[nodiscard]] inline torch::Tensor penalty(const MaxNormDescriptor& descriptor, const torch::Tensor& params)
    {
        const auto& options = descriptor.options;
        if (options.coefficient == 0.0 || params.numel() == 0) {
            return params.new_zeros({});
        }

        auto tensor = detail::move_dim_to_front(params, options.dim);
        if (tensor.dim() == 1) {
            tensor = tensor.view({tensor.size(0), 1});
        } else {
            tensor = tensor.flatten(1);
        }
        auto norms = torch::sqrt(tensor.pow(2).sum(1));
        auto excess = torch::relu(norms - options.max_norm);
        return excess.pow(2).sum().mul(options.coefficient);
    }

}

#endif // OMNI_REGULARIZATION_DETAILS_MAXNORM_HPP