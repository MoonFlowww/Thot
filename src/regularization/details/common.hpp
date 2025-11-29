#ifndef Nott_REGULARIZATION_DETAILS_COMMON_HPP
#define Nott_REGULARIZATION_DETAILS_COMMON_HPP

#include <torch/torch.h>

#include <cstdint>
#include <numeric>
#include <vector>

namespace Nott::Regularization::Details::detail {

    [[nodiscard]] inline torch::Tensor zeros_like_optional(const torch::Tensor& reference)
    {
        if (reference.defined()) {
            return reference.new_zeros({});
        }
        return torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32));
    }

    [[nodiscard]] inline torch::Tensor move_dim_to_front(const torch::Tensor& tensor, std::int64_t dim)
    {
        auto adjusted_dim = dim;
        if (adjusted_dim < 0) {
            adjusted_dim += tensor.dim();
        }
        if (adjusted_dim <= 0 || tensor.dim() <= 1) {
            return tensor;
        }

        std::vector<std::int64_t> permutation(tensor.dim());
        std::iota(permutation.begin(), permutation.end(), 0);
        std::swap(permutation[0], permutation[adjusted_dim]);
        return tensor.permute(permutation);
    }

}

#endif // Nott_REGULARIZATION_DETAILS_COMMON_HPP