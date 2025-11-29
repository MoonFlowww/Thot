#ifndef OMNI_NLL_HPP
#define OMNI_NLL_HPP

#include <optional>
#include <stdexcept>
#include <torch/torch.h>

#include "reduction.hpp"

namespace Omni::Loss::Details {
    struct NegativeLogLikelihoodOptions {
        Reduction reduction{Reduction::Mean};
        std::vector<double> weight{};
        std::optional<int64_t> ignore_index{};
    };

    struct NegativeLogLikelihoodDescriptor {
        NegativeLogLikelihoodOptions options{};
    };

    inline torch::Tensor compute(const NegativeLogLikelihoodDescriptor& descriptor, const torch::Tensor& prediction, const torch::Tensor& target, const std::optional<torch::Tensor>& weight = std::nullopt)
    {
        auto opts = torch::nn::functional::NLLLossFuncOptions{};
        opts = opts.reduction(to_torch_reduction<torch::nn::functional::NLLLossFuncOptions>(descriptor.options.reduction));

        if (!descriptor.options.weight.empty()) {
            auto weight_tensor = torch::tensor(
                descriptor.options.weight,
                torch::TensorOptions().dtype(prediction.scalar_type()).device(prediction.device()));
            opts = opts.weight(weight_tensor);
        } else if (weight && weight->defined()) {
            opts = opts.weight(weight->to(prediction.device(), prediction.scalar_type()));
        }

        if (descriptor.options.ignore_index.has_value()) {
            opts = opts.ignore_index(descriptor.options.ignore_index.value());
        }

        return torch::nn::functional::nll_loss(prediction, target, opts);
    }

}

#endif // OMNI_NLL_HPP