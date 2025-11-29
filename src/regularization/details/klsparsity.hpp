#ifndef Nott_REGULARIZATION_DETAILS_KLSPARSITY_HPP
#define Nott_REGULARIZATION_DETAILS_KLSPARSITY_HPP

#include <torch/torch.h>

namespace Nott::Regularization::Details {

    struct KLSparsityOptions {
        double coefficient{0.0};
        double target{0.05};
        double epsilon{1e-6};
    };

    struct KLSparsityDescriptor {
        KLSparsityOptions options{};
    };

    [[nodiscard]] inline torch::Tensor penalty(const KLSparsityDescriptor& descriptor,
                                               const torch::Tensor& activations)
    {
        const auto& options = descriptor.options;
        if (options.coefficient == 0.0 || activations.numel() == 0) {
            return activations.new_zeros({});
        }

        auto rho_hat = activations;
        if (rho_hat.dim() > 1) {
            rho_hat = rho_hat.mean(0);
        }

        auto clamp = torch::clamp(rho_hat, options.epsilon, 1.0 - options.epsilon);
        auto target_tensor = torch::full_like(clamp, options.target);
        auto one_minus_target = torch::full_like(clamp, 1.0 - options.target);
        auto one_minus_clamp = torch::clamp(1.0 - clamp, options.epsilon, 1.0);

        auto term = target_tensor * (torch::log(target_tensor) - torch::log(clamp))
            + one_minus_target * (torch::log(one_minus_target) - torch::log(one_minus_clamp));
        return term.sum().mul(options.coefficient);
    }

}

#endif // Nott_REGULARIZATION_DETAILS_KLSPARSITY_HPP