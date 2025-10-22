#ifndef THOT_TANH_HPP
#define THOT_TANH_HPP

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Thot::Activation::Details {

    struct Tanh {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            return torch::tanh(std::move(input));
        }
    };

}

#endif //THOT_TANH_HPP