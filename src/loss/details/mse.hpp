#ifndef THOT_MSE_HPP
#define THOT_MSE_HPP

#include <optional>
#include <stdexcept>
#include <torch/torch.h>
#include "reduction.hpp"


namespace Thot::Loss::Details {
    struct MSEOptions {
        Reduction reduction{Reduction::Mean};
        bool use_weight{false};
    };



    struct MSEDescriptor {
        MSEOptions options{};
    };

    namespace detail {
        inline torch::Tensor apply_reduction(torch::Tensor values, Reduction reduction) {
            switch (reduction) {
                case Reduction::Sum:
                    return values.sum();
                case Reduction::None:
                    return values;
                case Reduction::Mean:
                default:
                    return values.mean();
        }
    }
}

inline torch::Tensor compute(const MSEDescriptor& descriptor,
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    const std::optional<torch::Tensor>& weight = std::nullopt) {
        auto loss = torch::pow(prediction - target, 2);

        if (descriptor.options.use_weight) {
            if (!weight.has_value() || !weight->defined()) {
                throw std::invalid_argument(
                    "MSE loss configured to use weight but no weight tensor was provided.");
            }
            const auto weight_tensor = weight->to(prediction.device(), prediction.scalar_type());
            loss = loss * weight_tensor;
        }

        return detail::apply_reduction(std::move(loss), descriptor.options.reduction);
    }

}

#endif // THOT_MSE_HPP