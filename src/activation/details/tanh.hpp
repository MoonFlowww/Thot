#ifndef OMNI_TANH_HPP
#define OMNI_TANH_HPP

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Omni::Activation::Details {

    struct Tanh {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            return torch::tanh(std::move(input));
        }
    };

}

#endif //OMNI_TANH_HPP