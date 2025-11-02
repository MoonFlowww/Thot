#ifndef THOT_MAE_HPP
#define THOT_MAE_HPP
#include <optional>
#include <stdexcept>
#include <torch/torch.h>

#include "helper.hpp" // reduction called in

namespace Thot::Loss::Details {
    struct MAEOptions {
        Reduction reduction{Reduction::Mean};
        bool use_weight{false};
    };

    struct MAEDescriptor {
        MAEOptions options{};
    };

    inline torch::Tensor compute(const MAEDescriptor& descriptor, const torch::Tensor& prediction, const torch::Tensor& target, const std::optional<torch::Tensor>& weight = std::nullopt) {
        if (!descriptor.options.use_weight) {
            return torch::nn::functional::l1_loss(
                prediction,
                target,
                torch::nn::functional::L1LossFuncOptions().reduction(
                    torch::nn::functional::L1LossFuncOptions::reduction_t{to_torch_reduction<torch::nn::functional::L1LossFuncOptions>(descriptor.options.reduction)})
            );
        }

        if (!weight || !weight->defined()) {
            throw std::invalid_argument(
                "MeanAbsoluteError configured to use weight but no weight tensor was provided.");
        }

        auto per_elem = torch::nn::functional::l1_loss(
            prediction,
            target,
            torch::nn::functional::L1LossFuncOptions().reduction(torch::kNone)
        );

        return apply_reduction_weighted(per_elem, *weight, descriptor.options.reduction);
    }

}

#endif // THOT_MAE_HPP