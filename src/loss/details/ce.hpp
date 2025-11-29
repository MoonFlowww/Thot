#ifndef OMNI_CE_HPP
#define OMNI_CE_HPP
#include <optional>
#include <stdexcept>
#include <torch/torch.h>
#include <vector>

#include "reduction.hpp"

namespace Omni::Loss::Details {
    struct CrossEntropyOptions {
        Reduction reduction{Reduction::Mean};
        std::vector<double> weight{};
        double label_smoothing{0.0};
    };

    struct CrossEntropyDescriptor {
        CrossEntropyOptions options{};
    };

    inline torch::Tensor compute(const CrossEntropyDescriptor& descriptor,
                                 const torch::Tensor& prediction,
                                 const torch::Tensor& target,
                                 const std::optional<torch::Tensor>& weight = std::nullopt) {
        auto opts = torch::nn::functional::CrossEntropyFuncOptions{};
        opts = opts.reduction(to_torch_reduction<torch::nn::functional::CrossEntropyFuncOptions>(descriptor.options.reduction));
        opts = opts.label_smoothing(descriptor.options.label_smoothing);
        if (!descriptor.options.weight.empty()) {
            auto weight_tensor = torch::tensor(
                descriptor.options.weight,
                torch::TensorOptions().dtype(prediction.scalar_type()).device(prediction.device()));
            opts = opts.weight(weight_tensor);
        } else if (weight.has_value() && weight->defined()) {
            opts = opts.weight(weight->to(prediction.device(), prediction.scalar_type()));
        }
        return torch::nn::functional::cross_entropy(prediction, target, opts);
    }
}
#endif //OMNI_CE_HPP