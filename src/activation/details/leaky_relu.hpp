#ifndef THOT_LEAKY_RELU_HPP
#define THOT_LEAKY_RELU_HPP

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Thot::Activation::Details {

    struct LeakyReLU {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            return torch::leaky_relu(std::move(input));
        }
    };

}

#endif //THOT_LEAKY_RELU_HPP