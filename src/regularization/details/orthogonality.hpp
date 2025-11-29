#ifndef Nott_REGULARIZATION_DETAILS_ORTHOGONALITY_HPP
#define Nott_REGULARIZATION_DETAILS_ORTHOGONALITY_HPP

#include <torch/torch.h>

namespace Nott::Regularization::Details {

    struct OrthogonalityOptions {
        double coefficient{0.0};
    };

    struct OrthogonalityDescriptor {
        OrthogonalityOptions options{};
    };

    [[nodiscard]] inline torch::Tensor penalty(const OrthogonalityDescriptor& descriptor, const torch::Tensor& params)
    {
        const auto& options = descriptor.options;
        if (options.coefficient == 0.0 || params.numel() == 0) {
            return params.new_zeros({});
        }

        auto tensor = params;
        if (tensor.dim() < 2) {
            tensor = tensor.view({tensor.numel(), 1});
        } else {
            tensor = tensor.flatten(1);
        }

        auto gram = torch::matmul(tensor.transpose(0, 1), tensor);
        auto identity = torch::eye(gram.size(0), gram.options());
        auto diff = gram - identity;
        return diff.pow(2).sum().mul(options.coefficient);
    }

}

#endif // Nott_REGULARIZATION_DETAILS_ORTHOGONALITY_HPP