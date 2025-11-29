#ifndef Nott_LEAKY_RELU_HPP
#define Nott_LEAKY_RELU_HPP

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Nott::Activation::Details {

    struct LeakyReLU {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            return torch::leaky_relu(std::move(input));
        }
    };

}

#endif //Nott_LEAKY_RELU_HPP