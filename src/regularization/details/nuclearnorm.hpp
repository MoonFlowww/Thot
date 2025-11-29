#ifndef Nott_NUCLEARNORM_HPP
#define Nott_NUCLEARNORM_HPP

#include <torch/torch.h>

namespace Nott::Regularization::Details {

    struct NuclearNormOptions {
        double strength{0.0};
    };

    struct NuclearNormDescriptor {
        NuclearNormOptions options{};
    };

    [[nodiscard]] inline torch::Tensor penalty(const NuclearNormDescriptor& descriptor,
                                               const torch::Tensor& params) {
        const auto& options = descriptor.options;
        if (options.strength == 0.0) {
            return params.new_zeros({});
        }

        auto tensor = params;
        if (tensor.dim() < 2) {
            tensor = tensor.view({1, -1});
        } else if (tensor.dim() > 2) {
            tensor = tensor.flatten(0, tensor.dim() - 2);
        }
        auto singular_values = torch::linalg_svdvals(tensor);
        auto norm = singular_values.sum();
        return norm.mul(options.strength);
    }

}

#endif //Nott_NUCLEARNORM_HPP