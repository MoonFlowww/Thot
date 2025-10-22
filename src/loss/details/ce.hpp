#ifndef THOT_CE_HPP
#define THOT_CE_HPP
#include <optional>
#include <stdexcept>
#include <torch/torch.h>

#include "reduction.hpp"

namespace Thot::Loss::Details {
    struct CrossEntropyOptions {
        Reduction reduction{Reduction::Mean};
        bool use_weight{false};
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
        opts = opts.reduction(to_torch_reduction(descriptor.options.reduction));
        opts = opts.label_smoothing(descriptor.options.label_smoothing);
        if (descriptor.options.use_weight) {
            if (!weight.has_value() || !weight->defined()) {
                throw std::invalid_argument("CrossEntropy loss configured to use weight but no weight tensor was provided.");
            }
            opts = opts.weight(*weight);
        }
        return torch::nn::functional::cross_entropy(prediction, target, opts);
    }
}
#endif //THOT_CE_HPP