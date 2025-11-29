#ifndef OMNI_LEAKY_RELU_HPP
#define OMNI_LEAKY_RELU_HPP

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Omni::Activation::Details {

    struct LeakyReLU {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            return torch::leaky_relu(std::move(input));
        }
    };

}

#endif //OMNI_LEAKY_RELU_HPP