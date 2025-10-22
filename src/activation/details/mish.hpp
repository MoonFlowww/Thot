#ifndef THOT_MISH_HPP
#define THOT_MISH_HPP

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Thot::Activation::Details {

    struct Mish {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            auto tensor = std::move(input);
            return tensor * torch::tanh(torch::softplus(tensor));
        }
    };

}

#endif //THOT_MISH_HPP