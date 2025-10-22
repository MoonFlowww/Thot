#ifndef THOT_SWISH_HPP
#define THOT_SWISH_HPP

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Thot::Activation::Details {

    struct Swish {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            auto tensor = std::move(input);
            return tensor * torch::sigmoid(tensor);
        }
    };

}

#endif //THOT_SWISH_HPP