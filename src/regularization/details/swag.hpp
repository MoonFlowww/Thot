#ifndef THOT_SWAG_HPP
#define THOT_SWAG_HPP

#include <torch/torch.h>

namespace Thot::Regularization::Details {

    struct SWAGOptions {
        double coefficient{0.0};
        double variance_epsilon{1e-8};
    };

    struct SWAGDescriptor {
        SWAGOptions options{};
    };

    struct SWAGState {
        torch::Tensor mean;
        torch::Tensor variance;
    };

    [[nodiscard]] inline torch::Tensor penalty(const SWAGDescriptor& descriptor,
                                               const torch::Tensor& params,
                                               const SWAGState& state) {
        const auto& options = descriptor.options;
        if (options.coefficient == 0.0 || !state.mean.defined() || !state.variance.defined()) {
            return params.new_zeros({});
        }

        auto mean = state.mean.to(params.device(), params.scalar_type());
        auto variance = state.variance.to(params.device(), params.scalar_type());

        TORCH_CHECK(mean.sizes() == params.sizes(), "SWAG mean tensor must match parameter shape.");
        TORCH_CHECK(variance.sizes() == params.sizes(), "SWAG variance tensor must match parameter shape.");

        auto diff = params - mean;
        auto safe_variance = variance + options.variance_epsilon;
        auto scaled = diff.pow(2) / safe_variance;
        return scaled.mean().mul(options.coefficient);
    }

}

#endif // THOT_SWAG_HPP