#ifndef OMNI_LOSS_TVERSKY_HPP
#define OMNI_LOSS_TVERSKY_HPP

#include <optional>
#include <stdexcept>
#include <torch/torch.h>

#include "reduction.hpp"

namespace Omni::Loss::Details {

    struct TverskyOptions {
        Reduction reduction{Reduction::Mean};
        double alpha{0.5};
        double beta{0.5};
        double smooth{1.0};
    };

    struct TverskyDescriptor {
        TverskyOptions options{};
    };

    inline torch::Tensor compute(const TverskyDescriptor& descriptor,
                                 const torch::Tensor& prediction,
                                 const torch::Tensor& target,
                                 const std::optional<torch::Tensor>& weight = std::nullopt)
    {
        if (!prediction.defined() || !target.defined()) {
            throw std::invalid_argument("Tversky loss requires defined prediction and target tensors.");
        }

        auto pred = prediction;
        auto tgt = target.to(pred.device(), pred.scalar_type());

        if (tgt.sizes() != pred.sizes()) {
            tgt = tgt.expand_as(pred);
        }

        if (pred.dim() == 0) {
            throw std::invalid_argument("Tversky loss expects tensors with at least one dimension.");
        }

        auto pred_flat = pred.contiguous().reshape({pred.size(0), -1});
        auto target_flat = tgt.contiguous().reshape({tgt.size(0), -1});

        auto true_positive = (pred_flat * target_flat).sum(1);
        auto false_positive = (pred_flat * (1.0 - target_flat)).sum(1);
        auto false_negative = ((1.0 - pred_flat) * target_flat).sum(1);

        auto numerator = true_positive + descriptor.options.smooth;
        auto denominator = true_positive
            + descriptor.options.alpha * false_positive
            + descriptor.options.beta * false_negative
            + descriptor.options.smooth;

        auto tversky_index = numerator / denominator.clamp_min(1e-12);
        auto tversky_loss = 1.0 - tversky_index;

        if (weight && weight->defined()) {
            return apply_reduction_weighted(tversky_loss, *weight, descriptor.options.reduction);
        }
        return apply_reduction(tversky_loss, descriptor.options.reduction);
    }

}

#endif // OMNI_LOSS_TVERSKY_HPP