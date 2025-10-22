#ifndef THOT_REDUCTION_HPP
#define THOT_REDUCTION_HPP
#include <torch/torch.h>
#include <ATen/core/Reduction.h>
namespace Thot::Loss::Details {
    enum class Reduction { Mean, Sum, None };

    inline constexpr torch::nn::functional::CrossEntropyFuncOptions::reduction_t to_torch_reduction(Reduction r) {
        using ReductionOption = torch::nn::functional::CrossEntropyFuncOptions::reduction_t;
        switch (r) {
            case Reduction::Sum:
                return ReductionOption{torch::kSum};
            case Reduction::None:
                return ReductionOption{torch::kNone};
            case Reduction::Mean:
            default:
                return ReductionOption{torch::kMean};
        }
    }
}
#endif // THOT_REDUCTION_HPP
