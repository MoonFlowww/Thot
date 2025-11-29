#ifndef Nott_SFGE_HPP
#define Nott_SFGE_HPP

#include <torch/torch.h>

#include <numeric>
#include <vector>

namespace Nott::Regularization::Details {

    struct SFGEOptions {
        double coefficient{0.0};
    };

    struct SFGEDescriptor {
        SFGEOptions options{};
    };

    struct SFGEState {
        std::vector<torch::Tensor> snapshots;
        std::vector<double> weights;
    };

    [[nodiscard]] inline torch::Tensor penalty(const SFGEDescriptor&, const torch::Tensor& params) {
        return params.new_zeros({});
    }

    [[nodiscard]] inline torch::Tensor penalty(const SFGEDescriptor& descriptor,
                                               const torch::Tensor& params,
                                               const SFGEState& state) {
        const auto& options = descriptor.options;
        if (options.coefficient == 0.0 || state.snapshots.empty()) {
            return params.new_zeros({});
        }

        TORCH_CHECK(state.weights.empty() || state.weights.size() == state.snapshots.size(),
                    "SFGE weights size must match snapshot count when provided.");

        torch::Tensor total = params.new_zeros({});
        double weight_sum = 0.0;

        for (std::size_t index = 0; index < state.snapshots.size(); ++index) {
            const auto& snapshot = state.snapshots[index];
            TORCH_CHECK(snapshot.defined(), "SFGE snapshots must be defined tensors.");
            auto converted = snapshot.to(params.device(), params.scalar_type());
            TORCH_CHECK(converted.sizes() == params.sizes(), "SFGE snapshot must match parameter shape.");

            const double weight = state.weights.empty() ? 1.0 : state.weights[index];
            weight_sum += weight;

            auto diff = params - converted;
            total = total + diff.pow(2).mean().mul(weight);
        }

        if (weight_sum == 0.0) {
            weight_sum = static_cast<double>(state.snapshots.size());
        }

        return total.mul(options.coefficient / weight_sum);
    }

}

#endif // Nott_SFGE_HPP
