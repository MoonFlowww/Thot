#ifndef OMNI_GELU_HPP
#define OMNI_GELU_HPP
// "Gaussian Error Linear Units (GELUs)" https://arxiv.org/pdf/1606.08415
#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"
namespace Omni::Activation::Details {

    struct GeLU {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            return torch::gelu(std::move(input));
        }
    };

}

#endif //OMNI_GELU_HPP