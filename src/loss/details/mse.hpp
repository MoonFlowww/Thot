#ifndef THOT_MSE_HPP
#define THOT_MSE_HPP

#include <optional>
#include <stdexcept>
#include <torch/torch.h>
#include "reduction.hpp"

namespace Thot::Loss::Details {

    namespace F = torch::nn::functional;

    struct MSEOptions {
        Reduction reduction{Reduction::Mean};
        bool use_weight{false};
    };

    struct MSEDescriptor {
        MSEOptions options{};
    };

    // Map to the *MSE* functional reduction enum (unique name to avoid collisions)
    inline F::MSELossFuncOptions::reduction_t mse_reduction(Reduction r) {
        switch (r) {
            case Reduction::Sum:  return torch::kSum;
            case Reduction::None: return torch::kNone;
            case Reduction::Mean:
            default:              return torch::kMean;
        }
    }

    // Weighted reduction (proper weighted-mean)
    inline torch::Tensor apply_reduction_weighted(torch::Tensor loss,
                                                  const torch::Tensor& weight,
                                                  Reduction reduction) {
        // Make weight match loss device/dtype and broadcast if needed
        auto w = weight.to(loss.options()).expand_as(loss);

        switch (reduction) {
            case Reduction::None:
                return loss * w;
            case Reduction::Sum:
                return (loss * w).sum();
            case Reduction::Mean:
            default: {
                auto num = (loss * w).sum();
                auto den = w.sum().clamp_min(1e-12);
                return num / den;
            }
        }
    }

    inline torch::Tensor compute(const MSEDescriptor& descriptor,
                                 const torch::Tensor& prediction,
                                 const torch::Tensor& target,
                                 const std::optional<torch::Tensor>& weight = std::nullopt)
    {
        if (!descriptor.options.use_weight) {
            // Pure LibTorch MSE
            return F::mse_loss(
                prediction,
                target,
                F::MSELossFuncOptions().reduction(mse_reduction(descriptor.options.reduction))
            );
        }

        // Weighted path: get per-element loss, then reduce manually
        if (!weight || !weight->defined()) {
            throw std::invalid_argument(
                "MSE configured to use weight but no weight tensor was provided.");
        }

        auto per_elem = F::mse_loss(
            prediction,
            target,
            F::MSELossFuncOptions().reduction(torch::kNone)
        );

        return apply_reduction_weighted(per_elem, *weight, descriptor.options.reduction);
    }

} // namespace Thot::Loss::Details

#endif // THOT_MSE_HPP
