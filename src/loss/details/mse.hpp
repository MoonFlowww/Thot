#ifndef THOT_MSE_HPP
#define THOT_MSE_HPP

#include <optional>
#include <stdexcept>
#include <torch/torch.h>
#include "reduction.hpp"
namespace Thot::Loss::Details {


    namespace Thot::Loss::Details {

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
                                 const std::optional<torch::Tensor>& weight = std::nullopt) {
        auto opts = torch::nn::functional::MSELossFuncOptions{};
        opts = opts.reduction(to_torch_reduction(descriptor.options.reduction));
        if (descriptor.options.use_weight) {
            if (!weight.has_value() || !weight->defined()) {
                throw std::invalid_argument("MSE loss configured to use weight but no weight tensor was provided.");
            }
            opts = opts.weight(*weight);
        }
        return torch::nn::functional::mse_loss(prediction, target, opts);
    }
}

#endif //THOT_MSE_HPP
