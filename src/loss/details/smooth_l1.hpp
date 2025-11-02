#ifndef THOT_SMOOTH_L1_HPP
#define THOT_SMOOTH_L1_HPP

#include <optional>
#include <stdexcept>
#include <torch/torch.h>

#include "helper.hpp"

namespace Thot::Loss::Details {
    struct SmoothL1Options {
        Reduction reduction{Reduction::Mean};
        bool use_weight{false};
        double beta{1.0};
    };

    struct SmoothL1Descriptor {
        SmoothL1Options options{};
    };


    inline torch::Tensor compute(const SmoothL1Descriptor& descriptor, const torch::Tensor& prediction, const torch::Tensor& target, const std::optional<torch::Tensor>& weight = std::nullopt) {
        auto opts = torch::nn::functional::SmoothL1LossFuncOptions{};
        opts = opts.reduction(to_torch_reduction<torch::nn::functional::SmoothL1LossFuncOptions>(descriptor.options.reduction));
        opts = opts.beta(descriptor.options.beta);

        if (!descriptor.options.use_weight) {
            return torch::nn::functional::smooth_l1_loss(prediction, target, opts);
        }

        if (!weight || !weight->defined()) {
            throw std::invalid_argument(
                "SmoothL1 configured to use weight but no weight tensor was provided.");
        }

        auto per_elem = torch::nn::functional::smooth_l1_loss(
            prediction,
            target,
            torch::nn::functional::SmoothL1LossFuncOptions{}.reduction(torch::kNone).beta(descriptor.options.beta)
        );

        return apply_reduction_weighted(per_elem, *weight, descriptor.options.reduction);
    }

}

#endif // THOT_SMOOTH_L1_HPP