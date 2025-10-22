#ifndef THOT_MSE_HPP
#define THOT_MSE_HPP

#include <optional>
#include <stdexcept>
#include <torch/torch.h>

namespace Thot::Loss::Details {

enum class Reduction {
    Mean,
    Sum,
    None,
};

inline constexpr torch::Reduction to_torch_reduction(Reduction reduction) {
    switch (reduction) {
        case Reduction::Sum:
            return torch::kSum;
        case Reduction::None:
            return torch::kNone;
        case Reduction::Mean:
        default:
            return torch::kMean;
    }
}

struct MSEOptions {
    Reduction reduction{Reduction::Mean};
    bool use_weight{false};
};

struct MSEDescriptor {
    MSEOptions options{};
};

inline torch::Tensor compute(const MSEDescriptor& descriptor,
                             const torch::Tensor& prediction,
                             const torch::Tensor& target,
                             const std::optional<torch::Tensor>& weight = std::nullopt) {
    auto opts = torch::nn::functional::MSELossFuncOptions{};
    opts = opts.reduction(to_torch_reduction(descriptor.options.reduction));
    if (descriptor.options.use_weight) {
        if (!weight.has_value() || !weight->defined()) {
            throw std::invalid_argument("MSE loss configured to use weight but no weight tensor was provided.");
        }
        opts = opts.weight(*weight);
    }
    return torch::nn::functional::mse_loss(prediction, target, opts);
}

}  // namespace Thot::Loss::Details

#endif //THOT_MSE_HPP
