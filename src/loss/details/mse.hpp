#ifndef THOT_MSE_HPP
#define THOT_MSE_HPP

#include <optional>
#include <stdexcept>
#include <torch/torch.h>
#include "helper.hpp"

namespace Thot::Loss::Details {

    namespace F = torch::nn::functional;

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
                                 const std::optional<torch::Tensor>& weight = std::nullopt)
    {
        if (!descriptor.options.use_weight) {
            // Pure LibTorch MSE
            return F::mse_loss(
                prediction,
                target,
                torch::nn::functional::MSELossFuncOptions().reduction(to_torch_reduction<torch::nn::functional::MSELossFuncOptions>(descriptor.options.reduction))
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
