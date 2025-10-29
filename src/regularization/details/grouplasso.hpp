#ifndef THOT_REGULARIZATION_DETAILS_GROUPLASSO_HPP
#define THOT_REGULARIZATION_DETAILS_GROUPLASSO_HPP

#include <torch/torch.h>

#include "common.hpp"

namespace Thot::Regularization::Details {

    struct GroupLassoOptions {
        double coefficient{0.0};
        std::int64_t group_dim{0};
        double epsilon{1e-8};
    };

    struct GroupLassoDescriptor {
        GroupLassoOptions options{};
    };

    [[nodiscard]] inline torch::Tensor penalty(const GroupLassoDescriptor& descriptor, const torch::Tensor& params)
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
        auto norms = torch::sqrt(squared + options.epsilon);
        return norms.sum().mul(options.coefficient);
    }

} // namespace Thot::Regularization::Details

#endif // THOT_REGULARIZATION_DETAILS_GROUPLASSO_HPP