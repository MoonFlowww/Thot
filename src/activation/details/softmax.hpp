#ifndef Nott_SOFTMAX_HPP
#define Nott_SOFTMAX_HPP

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Nott::Activation::Details {

    struct Softmax {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            if (input.dim() == 0) {
                return input;
            }
            const auto dim = input.dim() - 1;
            return torch::softmax(std::move(input), dim);
        }
    };

}

#endif //Nott_SOFTMAX_HPP