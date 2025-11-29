#ifndef OMNI_SIGMOID_HPP
#define OMNI_SIGMOID_HPP

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Omni::Activation::Details {
    struct Sigmoid {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            return torch::sigmoid(std::move(input));
        }
    };
}

#endif //OMNI_SIGMOID_HPP