#ifndef THOT_LOSS_KL_HPP
#define THOT_LOSS_KL_HPP
#include <torch/torch.h>

#include "reduction.hpp"

namespace Thot::Loss::Details {
    struct KLDivOptions {
        Reduction reduction{Reduction::Mean};
        bool log_target{false};
        bool use_batch_mean{false};
    };

    struct KLDivDescriptor {
        KLDivOptions options{};
    };

    inline torch::Tensor compute(const KLDivDescriptor& descriptor, const torch::Tensor& prediction, const torch::Tensor& target, int64_t dim = 1, bool prediction_is_log = false)  {
        torch::Tensor input = prediction;
        if (!prediction_is_log) {
            input = torch::nn::functional::log_softmax(prediction, torch::nn::functional::LogSoftmaxFuncOptions(dim));
        }

        // Ensure target is floating and on same device/dtype as input.
        torch::Tensor tgt = target.to(input.dtype()).to(input.device());

        // If target is given as class indices (int64), convert to one-hot prob
        if (tgt.dtype() == torch::kLong) {
            const auto num_classes = input.size(dim);
            tgt = torch::nn::functional::one_hot(tgt, num_classes).to(input.dtype());
            tgt = tgt / tgt.sum(dim, /*keepdim=*/true).clamp_min(1e-12);
        }

        auto opts = torch::nn::functional::KLDivFuncOptions{};
        if (descriptor.options.use_batch_mean) {
            opts = opts.reduction(torch::kBatchMean);
        } else {
            opts = opts.reduction(to_torch_reduction<torch::nn::functional::KLDivFuncOptions>(descriptor.options.reduction));
        }
        opts = opts.log_target(descriptor.options.log_target);

        return torch::nn::functional::kl_div(input, tgt, opts);
    }
}
#endif //THOT_LOSS_KL_HPP