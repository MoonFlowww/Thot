#ifndef THOT_MARGIN_RANKING_HPP
#define THOT_MARGIN_RANKING_HPP

#include <torch/torch.h>

#include "reduction.hpp"

namespace Thot::Loss::Details {
    struct MarginRankingOptions {
        Reduction reduction{Reduction::Mean};
        double margin{0.0};
    };

    struct MarginRankingDescriptor {
        MarginRankingOptions options{};
    };

    inline torch::Tensor compute(const MarginRankingDescriptor& descriptor, const torch::Tensor& input1, const torch::Tensor& input2, const torch::Tensor& target) {
        auto opts = torch::nn::functional::MarginRankingLossFuncOptions{};
        opts = opts.margin(descriptor.options.margin);
        opts = opts.reduction(torch::nn::functional::MarginRankingLossFuncOptions::reduction_t{to_torch_reduction<torch::nn::functional::MarginRankingLossFuncOptions>(descriptor.options.reduction)});

        return torch::nn::functional::margin_ranking_loss(input1, input2, target, opts);
    }

}

#endif // THOT_MARGIN_RANKING_HPP