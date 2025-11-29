#ifndef Nott_TANH_HPP
#define Nott_TANH_HPP

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Nott::Activation::Details {

    struct Tanh {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            return torch::tanh(std::move(input));
        }
    };

}

#endif //Nott_TANH_HPP