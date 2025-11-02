#ifndef THOT_BCE_HPP
#define THOT_BCE_HPP

#include <optional>
#include <stdexcept>
#include <torch/torch.h>

#include "reduction.hpp"

namespace Thot::Loss::Details {

    struct BCEWithLogitsOptions {
        Reduction reduction{Reduction::Mean};
        bool use_weight{false};
        bool use_pos_weight{false};
    };

    struct BCEWithLogitsDescriptor {
        BCEWithLogitsOptions options{};
    };

    inline torch::Tensor compute(const BCEWithLogitsDescriptor& descriptor, const torch::Tensor& prediction, const torch::Tensor& target,
                                 const std::optional<torch::Tensor>& weight = std::nullopt, const std::optional<torch::Tensor>& pos_weight = std::nullopt) {
        auto opts = torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions{};
        opts = opts.reduction(to_torch_reduction<torch::nn::functional::BinaryCrossEntropyFuncOptions>(descriptor.options.reduction));

        if (descriptor.options.use_weight) {
            if (!weight || !weight->defined()) {
                throw std::invalid_argument(
                    "BinaryCrossEntropyWithLogits configured to use weight but no weight tensor was provided.");
            }
            opts = opts.weight(weight->to(prediction.device(), prediction.scalar_type()));
        }

        if (descriptor.options.use_pos_weight) {
            if (!pos_weight || !pos_weight->defined()) {
                throw std::invalid_argument(
                    "BinaryCrossEntropyWithLogits configured to use pos_weight but no pos_weight tensor was provided.");
            }
            opts = opts.pos_weight(pos_weight->to(prediction.device(), prediction.scalar_type()));
        }

        return torch::nn::functional::binary_cross_entropy_with_logits(prediction, target, opts);
    }

}

#endif // THOT_BCE_HPP