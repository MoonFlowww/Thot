#ifndef OMNI_REGULARIZATION_DETAILS_STRUCTURED_L2_HPP
#define OMNI_REGULARIZATION_DETAILS_STRUCTURED_L2_HPP

#include <torch/torch.h>

#include "common.hpp"

namespace Omni::Regularization::Details {

    struct StructuredL2Options {
        double coefficient{0.0};
        std::int64_t group_dim{0};
    };

    struct StructuredL2Descriptor {
        StructuredL2Options options{};
    };

    [[nodiscard]] inline torch::Tensor penalty(const StructuredL2Descriptor& descriptor, const torch::Tensor& params)
    {
        const auto& options = descriptor.options;
        if (options.coefficient == 0.0 || params.numel() == 0) {
            return params.new_zeros({});
        }

        auto tensor = detail::move_dim_to_front(params, options.group_dim);
        if (tensor.dim() == 0) {
            return params.new_zeros({});
        }

        if (tensor.dim() == 1) {
            tensor = tensor.view({tensor.size(0), 1});
        } else {
            tensor = tensor.flatten(1);
        }
        auto squared = tensor.pow(2).sum(1);
        return squared.sum().mul(options.coefficient);
    }

}

#endif // OMNI_REGULARIZATION_DETAILS_STRUCTURED_L2_HPP