#ifndef OMNI_DATA_TRANSFORM_DIMENSIONALITY_REDUCTION_COMMON_HPP
#define OMNI_DATA_TRANSFORM_DIMENSIONALITY_REDUCTION_COMMON_HPP

#include <cmath>
#include <cstddef>
#include <limits>

#include <torch/torch.h>

namespace Omni::Data::Transform::DimensionalityReduction {
    namespace Details {
        inline torch::Tensor shrinkage(const torch::Tensor& tensor, double tau) {
            if (tau <= 0.0) {
                return tensor;
            }
            auto abs_tensor = tensor.abs();
            auto sign_tensor = torch::sign(tensor);
            auto clipped = torch::relu(abs_tensor - tau);
            return sign_tensor * clipped;
        }

        inline double safe_norm(const torch::Tensor& tensor, double default_value = 1.0) {
            if (tensor.numel() == 0) {
                return default_value;
            }
            auto value = tensor.norm().item<double>();
            if (!std::isfinite(value) || value <= 0.0) {
                return default_value;
            }
            return value;
        }

        inline double spectral_norm(const torch::Tensor& tensor) {
            auto svd = torch::linalg_svdvals(tensor);
            if (svd.numel() == 0) {
                return 0.0;
            }
            return svd.max().item<double>();
        }
    }
}

#endif // OMNI_DATA_TRANSFORM_DIMENSIONALITY_REDUCTION_COMMON_HPP