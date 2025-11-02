#ifndef THOT_NLL_HPP
#define THOT_NLL_HPP

#include <optional>
#include <stdexcept>
#include <torch/torch.h>

#include "reduction.hpp"

namespace Thot::Loss::Details {
    struct NegativeLogLikelihoodOptions {
        Reduction reduction{Reduction::Mean};
        bool use_weight{false};
        std::optional<int64_t> ignore_index{};
    };

    struct NegativeLogLikelihoodDescriptor {
        NegativeLogLikelihoodOptions options{};
    };

    inline torch::Tensor compute(const NegativeLogLikelihoodDescriptor& descriptor, const torch::Tensor& prediction, const torch::Tensor& target, const std::optional<torch::Tensor>& weight = std::nullopt)
    {
        auto opts = torch::nn::functional::NLLLossFuncOptions{};
        opts = opts.reduction(to_torch_reduction<torch::nn::functional::NLLLossFuncOptions>(descriptor.options.reduction));

        if (descriptor.options.use_weight) {
            if (!weight || !weight->defined()) {
                throw std::invalid_argument(
                    "NegativeLogLikelihood configured to use weight but no weight tensor was provided.");
            }
            opts = opts.weight(weight->to(prediction.device(), prediction.scalar_type()));
        }

        if (descriptor.options.ignore_index.has_value()) {
            opts = opts.ignore_index(descriptor.options.ignore_index.value());
        }

        return torch::nn::functional::nll_loss(prediction, target, opts);
    }

}

#endif // THOT_NLL_HPP