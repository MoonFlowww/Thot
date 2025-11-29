#ifndef OMNI_SMOOTH_L1_HPP
#define OMNI_SMOOTH_L1_HPP

#include <optional>
#include <stdexcept>
#include <torch/torch.h>

#include "helper.hpp"

namespace Omni::Loss::Details {
    struct SmoothL1Options {
        Reduction reduction{Reduction::Mean};
        std::vector<double> weight{};
        double beta{1.0};
    };

    struct SmoothL1Descriptor {
        SmoothL1Options options{};
    };


    inline torch::Tensor compute(const SmoothL1Descriptor& descriptor, const torch::Tensor& prediction, const torch::Tensor& target, const std::optional<torch::Tensor>& weight = std::nullopt) {
        auto opts = torch::nn::functional::SmoothL1LossFuncOptions{};
        opts = opts.reduction(to_torch_reduction<torch::nn::functional::SmoothL1LossFuncOptions>(descriptor.options.reduction));
        opts = opts.beta(descriptor.options.beta);


        auto per_elem = torch::nn::functional::smooth_l1_loss(
            prediction,
            target,
            torch::nn::functional::SmoothL1LossFuncOptions{}.reduction(torch::kNone).beta(descriptor.options.beta)
        );

        std::optional<torch::Tensor> weight_tensor{};
        if (!descriptor.options.weight.empty()) {
            auto tensor = torch::tensor(
                descriptor.options.weight,
                torch::TensorOptions().dtype(prediction.scalar_type()).device(prediction.device()));
            if (tensor.numel() == per_elem.numel()) {
                tensor = tensor.reshape(per_elem.sizes());
            }
            weight_tensor = tensor;
        } else if (weight && weight->defined()) {
            weight_tensor = weight->to(prediction.device(), prediction.scalar_type());
        }

        if (!weight_tensor) {
            return torch::nn::functional::smooth_l1_loss(prediction, target, opts);
        }

        return apply_reduction_weighted(per_elem, *weight_tensor, descriptor.options.reduction);
    }

}

#endif // OMNI_SMOOTH_L1_HPP