#ifndef THOT_REDUCTION_HPP
#define THOT_REDUCTION_HPP
#include <torch/torch.h>
#include <ATen/core/Reduction.h>
namespace Thot::Loss::Details {
    enum class Reduction { Mean, Sum, None };

    inline constexpr at::Reduction::Reduction to_torch_reduction(Reduction r) {
        using TorchReduction = at::Reduction::Reduction;
        switch (r) {
            case Reduction::Sum:  return TorchReduction::Sum;
            case Reduction::None: return TorchReduction::None;
            case Reduction::Mean:
            default:              return TorchReduction::Mean;
        }
    }
}
#endif // THOT_REDUCTION_HPP
