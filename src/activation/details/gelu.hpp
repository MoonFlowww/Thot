#ifndef THOT_GELU_HPP
#define THOT_GELU_HPP

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"
namespace Thot::Activation::Details {

    struct GeLU {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            return torch::gelu(std::move(input));
        }
    };

}

#endif //THOT_GELU_HPP