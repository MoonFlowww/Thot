#ifndef Nott_RELU_HPP
#define Nott_RELU_HPP

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Nott::Activation::Details {
    struct ReLU {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            return torch::relu(std::move(input));
        }
    };


}

#endif //Nott_RELU_HPP
