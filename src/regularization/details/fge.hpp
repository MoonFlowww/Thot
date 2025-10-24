#ifndef THOT_FGE_HPP
#define THOT_FGE_HPP

#include <torch/torch.h>

#include <vector>

namespace Thot::Regularization::Details {

    struct FGEOptions {
        double coefficient{0.0};
    };

    struct FGEDescriptor {
        FGEOptions options{};
    };

    struct FGEState {
        std::vector<torch::Tensor> snapshots;
    };

    [[nodiscard]] inline torch::Tensor penalty(const FGEDescriptor& descriptor,
                                               const torch::Tensor& params,
                                               const FGEState& state) {
        const auto& options = descriptor.options;
        if (options.coefficient == 0.0 || state.snapshots.empty()) {
            return params.new_zeros({});
        }

        torch::Tensor total = params.new_zeros({});
        const auto num_snapshots = static_cast<double>(state.snapshots.size());
        for (const auto& snapshot : state.snapshots) {
            TORCH_CHECK(snapshot.defined(), "FGE snapshots must be defined tensors.");
            auto converted = snapshot.to(params.device(), params.scalar_type());
            TORCH_CHECK(converted.sizes() == params.sizes(), "FGE snapshot must match parameter shape.");
            auto diff = params - converted;
            total = total + diff.pow(2).mean();
        }

        return total.mul(options.coefficient / num_snapshots);
    }

}

#endif // THOT_FGE_HPP