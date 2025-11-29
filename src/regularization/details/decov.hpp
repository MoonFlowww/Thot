#ifndef OMNI_REGULARIZATION_DETAILS_DECOV_HPP
#define OMNI_REGULARIZATION_DETAILS_DECOV_HPP

#include <torch/torch.h>

namespace Omni::Regularization::Details {

    struct DeCovOptions {
        double coefficient{0.0};
        double epsilon{1e-5};
    };

    struct DeCovDescriptor {
        DeCovOptions options{};
    };

    [[nodiscard]] inline torch::Tensor penalty(const DeCovDescriptor& descriptor, const torch::Tensor& activations)
    {
        const auto& options = descriptor.options;
        if (options.coefficient == 0.0 || activations.numel() == 0) {
            return activations.new_zeros({});
        }

        auto tensor = activations;
        if (tensor.dim() < 2) {
            tensor = tensor.view({1, tensor.numel()});
        } else {
            tensor = tensor.flatten(1);
        }

        if (tensor.size(0) <= 1) {
            return activations.new_zeros({});
        }

        auto centered = tensor - tensor.mean(0, true);
        auto covariance = torch::matmul(centered.transpose(0, 1), centered)
            / static_cast<double>(tensor.size(0) - 1);
        auto diag = torch::diag(covariance);
        auto off_diag = covariance - torch::diag(diag);
        auto squared = off_diag.pow(2).sum();
        return squared.mul(options.coefficient);
    }

}

#endif // OMNI_REGULARIZATION_DETAILS_DECOV_HPP
