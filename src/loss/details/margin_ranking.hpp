#ifndef THOT_MARGIN_RANKING_HPP
#define THOT_MARGIN_RANKING_HPP

#include <optional>
#include <stdexcept>
#include <vector>
#include <torch/torch.h>

#include "reduction.hpp"

namespace Thot::Loss::Details {
    struct MarginRankingOptions {
        Reduction reduction{Reduction::Mean};
        double margin{0.0};
        std::vector<double> weight{};
    };

    struct MarginRankingDescriptor {
        MarginRankingOptions options{};
    };

    inline torch::Tensor compute(const MarginRankingDescriptor& descriptor, const torch::Tensor& prediction, const torch::Tensor& target, const std::optional<torch::Tensor>& weight = std::nullopt) {


        TORCH_CHECK(prediction.dim() >= 2,
                    "MarginRanking expects a pair tensor with a dimension of size 2, got ", prediction.sizes());

        int64_t pair_axis = -1;
        if (prediction.size(1) == 2) {
            pair_axis = 1;
        } else if (prediction.size(0) == 2) {
            pair_axis = 0;
        } else {
            TORCH_CHECK(false,
                        "MarginRanking expects prediction shaped [B, 2, ...] or [2, B, ...], got ",
                        prediction.sizes());
        }

        auto input1 = prediction.select(pair_axis, 0).contiguous();
        auto input2 = prediction.select(pair_axis, 1).contiguous();
        auto opts = torch::nn::functional::MarginRankingLossFuncOptions{};
        opts = opts.margin(descriptor.options.margin);
        opts = opts.reduction(torch::nn::functional::MarginRankingLossFuncOptions::reduction_t{
            to_torch_reduction<torch::nn::functional::MarginRankingLossFuncOptions>(descriptor.options.reduction)
        });

        if (!descriptor.options.weight.empty()) {
            auto weight_tensor = torch::tensor(
                descriptor.options.weight,
                torch::TensorOptions().dtype(input1.scalar_type()).device(input1.device()));
            if (weight_tensor.numel() == input1.numel()) {
                weight_tensor = weight_tensor.reshape(input1.sizes());
            }
            opts = opts.weight(weight_tensor);
        } else if (weight && weight->defined()) {
            opts = opts.weight(weight->to(input1.device(), input1.scalar_type()));
        }


        auto y = target.to(input1.device(), input1.scalar_type());
        if (y.sizes() != input1.sizes()) {
            y = y.reshape(input1.sizes());
        }

        return torch::nn::functional::margin_ranking_loss(input1, input2, y, opts);
    }

}

#endif // THOT_MARGIN_RANKING_HPP