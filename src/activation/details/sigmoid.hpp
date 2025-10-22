#ifndef THOT_SIGMOID_HPP
#define THOT_SIGMOID_HPP

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Thot::Activation::Details {
    struct Sigmoid {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            return torch::sigmoid(std::move(input));
        }
    };
}

#endif //THOT_SIGMOID_HPP