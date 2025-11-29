#ifndef OMNI_BCE_HPP
#define OMNI_BCE_HPP

#include <optional>
#include <stdexcept>
#include <torch/torch.h>

#include "reduction.hpp"

namespace Omni::Loss::Details {

    struct BCEWithLogitsOptions {
        Reduction reduction{Reduction::Mean};
        std::vector<double> weight{};
        std::vector<double> pos_weight{};
    };

    struct BCEWithLogitsDescriptor {
        BCEWithLogitsOptions options{};
    };

    inline torch::Tensor compute(const BCEWithLogitsDescriptor& descriptor, const torch::Tensor& prediction, const torch::Tensor& target,
                                 const std::optional<torch::Tensor>& weight = std::nullopt, const std::optional<torch::Tensor>& pos_weight = std::nullopt) {
        auto opts = torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions{};
        opts = opts.reduction(to_torch_reduction<torch::nn::functional::BinaryCrossEntropyFuncOptions>(descriptor.options.reduction));

        if (!descriptor.options.weight.empty()) {
            auto weight_tensor = torch::tensor(
                descriptor.options.weight,
                torch::TensorOptions().dtype(prediction.scalar_type()).device(prediction.device()));
            opts = opts.weight(weight_tensor);
        } else if (weight && weight->defined()) {
            opts = opts.weight(weight->to(prediction.device(), prediction.scalar_type()));
        }

        if (!descriptor.options.pos_weight.empty()) {
            auto pos_weight_tensor = torch::tensor(
                descriptor.options.pos_weight,
                torch::TensorOptions().dtype(prediction.scalar_type()).device(prediction.device()));
            opts = opts.pos_weight(pos_weight_tensor);
        } else if (pos_weight && pos_weight->defined()) {
            opts = opts.pos_weight(pos_weight->to(prediction.device(), prediction.scalar_type()));
        }

        return torch::nn::functional::binary_cross_entropy_with_logits(prediction, target, opts);
    }

}

#endif // OMNI_BCE_HPP