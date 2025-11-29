#ifndef Nott_REGULARIZATION_DETAILS_WGANGP_HPP
#define Nott_REGULARIZATION_DETAILS_WGANGP_HPP

#include <torch/torch.h>

namespace Nott::Regularization::Details {

    struct WGANGPOptions {
        double coefficient{0.0};
        double target{1.0};
    };

    struct WGANGPDescriptor {
        WGANGPOptions options{};
    };

    [[nodiscard]] inline torch::Tensor penalty(const WGANGPDescriptor& descriptor, const torch::Tensor& gradients)
    {
        const auto& options = descriptor.options;
        if (options.coefficient == 0.0 || gradients.numel() == 0 || gradients.dim() == 0) {
            return gradients.new_zeros({});
        }

        auto tensor = gradients.view({gradients.size(0), -1});
        auto norms = tensor.norm(2, 1, false);
        auto penalty_value = (norms - options.target).pow(2).mean();
        return penalty_value.mul(options.coefficient);
    }

}

#endif // Nott_REGULARIZATION_DETAILS_WGANGP_HPP