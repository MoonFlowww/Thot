#ifndef THOT_RELU_HPP
#define THOT_RELU_HPP

#include <torch/torch.h>

#include <utility>

#include "activation/activation.hpp"

namespace Thot::Activation::Details {

struct ReLU {
    [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
        return torch::relu(std::move(input));
    }
};

inline torch::Tensor apply(::Thot::Activation::Type type, torch::Tensor input) {
    switch (type) {
        case ::Thot::Activation::Type::ReLU:
            return ReLU{}(std::move(input));
        case ::Thot::Activation::Type::Identity:
        default:
            return input;
    }
}

}  // namespace Thot::Activation::Details

#endif //THOT_RELU_HPP
