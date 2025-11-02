#ifndef THOT_LOSS_HELPER_HPP
#define THOT_LOSS_HELPER_HPP
#include "reduction.hpp"

namespace Thot::Loss::Details {
    inline torch::Tensor apply_reduction_weighted(torch::Tensor loss, const torch::Tensor& weight, Reduction reduction) {
        auto w = weight.to(loss.options()).expand_as(loss);

        switch (reduction) {
            case Reduction::None:
                return loss * w;
            case Reduction::Sum:
                return (loss * w).sum();
            case Reduction::Mean:
            default: {
                auto num = (loss * w).sum();
                auto den = w.sum().clamp_min(1e-12);
                return num / den;
            }
        }
    }
}
#endif //THOT_LOSS_HELPER_HPP