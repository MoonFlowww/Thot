#ifndef THOT_REDUCTION_HPP
#define THOT_REDUCTION_HPP
#include <torch/torch.h>
#include <ATen/core/Reduction.h>
namespace Thot::Loss::Details {
    enum class Reduction { Mean, Sum, None };

    inline constexpr at::Reduction to_torch_reduction(Reduction r) {
        switch (r) {
            case Reduction::Sum:  return at::Reduction::Sum;
            case Reduction::None: return at::Reduction::None;
            case Reduction::Mean:
            default:              return at::Reduction::Mean;
        }
    }
}
#endif // THOT_REDUCTION_HPP
