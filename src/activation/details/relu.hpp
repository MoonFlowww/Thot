#ifndef OMNI_RELU_HPP
#define OMNI_RELU_HPP

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Omni::Activation::Details {
    struct ReLU {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            return torch::relu(std::move(input));
        }
    };


}

#endif //OMNI_RELU_HPP
