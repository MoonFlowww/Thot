#ifndef Nott_SIGMOID_HPP
#define Nott_SIGMOID_HPP

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Nott::Activation::Details {
    struct Sigmoid {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            return torch::sigmoid(std::move(input));
        }
    };
}

#endif //Nott_SIGMOID_HPP