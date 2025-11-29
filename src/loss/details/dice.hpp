#ifndef OMNI_LOSS_DICE_HPP
#define OMNI_LOSS_DICE_HPP

#include <optional>
#include <stdexcept>
#include <torch/torch.h>

#include "reduction.hpp"

namespace Omni::Loss::Details {

    struct DiceOptions {
        Reduction reduction{Reduction::Mean};
        double smooth{1.0};
        double exponent{2.0};
        bool clamp_predictions{true};
    };

    struct DiceDescriptor {
        DiceOptions options{};
    };

    inline torch::Tensor compute(const DiceDescriptor& descriptor,
                                 const torch::Tensor& prediction,
                                 const torch::Tensor& target,
                                 const std::optional<torch::Tensor>& weight = std::nullopt)
    {
        if (!prediction.defined() || !target.defined()) {
            throw std::invalid_argument("Dice loss requires defined prediction and target tensors.");
        }

        auto pred = prediction;
        auto tgt = target.to(pred.device(), pred.scalar_type());

        if (descriptor.options.clamp_predictions) {
            pred = pred.clamp(0.0, 1.0);
        }

        if (tgt.sizes() != pred.sizes()) {
            tgt = tgt.expand_as(pred);
        }

        auto flatten = [](const torch::Tensor& tensor) {
            if (tensor.dim() <= 1) {
                return tensor;
            }
            return tensor.reshape({tensor.size(0), -1});
        };

        auto pred_flat = flatten(pred.contiguous());
        auto target_flat = flatten(tgt.contiguous());

        auto intersection = (pred_flat * target_flat).sum(1);
        auto pred_sum = pred_flat.pow(descriptor.options.exponent).sum(1);
        auto target_sum = target_flat.pow(descriptor.options.exponent).sum(1);

        auto numerator = (intersection * 2.0) + descriptor.options.smooth;
        auto denominator = pred_sum + target_sum + descriptor.options.smooth;
        auto dice = numerator / denominator.clamp_min(1e-12);
        auto dice_loss = 1.0 - dice;

        if (weight && weight->defined()) {
            return apply_reduction_weighted(dice_loss, *weight, descriptor.options.reduction);
        }
        return apply_reduction(dice_loss, descriptor.options.reduction);
    }

}

#endif // OMNI_LOSS_DICE_HPP