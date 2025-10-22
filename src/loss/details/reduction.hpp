#ifndef THOT_REDUCTION_HPP
#define THOT_REDUCTION_HPP
#include <torch/torch.h>

namespace Thot::Loss::Details {
    enum class Reduction {
        Mean,
        Sum,
        None,
    };

    inline constexpr torch::Reduction to_torch_reduction(Reduction reduction) {
        switch (reduction) {
            case Reduction::Sum:
                return torch::kSum;
            case Reduction::None:
                return torch::kNone;
            case Reduction::Mean:
            default:
                return torch::kMean;
        }
    }
}

#endif //THOT_REDUCTION_HPP