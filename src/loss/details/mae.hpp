#ifndef THOT_MAE_HPP
#define THOT_MAE_HPP
#include <optional>
#include <stdexcept>
#include <torch/torch.h>

#include "helper.hpp" // reduction called in

namespace Thot::Loss::Details {
    struct MAEOptions {
        Reduction reduction{Reduction::Mean};
        std::vector<double> weight{};
    };

    struct MAEDescriptor {
        MAEOptions options{};
    };

    inline torch::Tensor compute(const MAEDescriptor& descriptor, const torch::Tensor& prediction, const torch::Tensor& target, const std::optional<torch::Tensor>& weight = std::nullopt) {
        auto per_elem = torch::nn::functional::l1_loss(
            prediction,
            target,
            torch::nn::functional::L1LossFuncOptions().reduction(torch::kNone)
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
            return torch::nn::functional::l1_loss(prediction, target,
                torch::nn::functional::L1LossFuncOptions().reduction(
                torch::nn::functional::L1LossFuncOptions::reduction_t{
                    to_torch_reduction<torch::nn::functional::L1LossFuncOptions>(descriptor.options.reduction)})
            );
        }

        return apply_reduction_weighted(per_elem, *weight_tensor, descriptor.options.reduction);
    }

}

#endif // THOT_MAE_HPP