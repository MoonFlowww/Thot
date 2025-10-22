#ifndef THOT_SILU_HPP
#define THOT_SILU_HPP

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Thot::Activation::Details {

    struct SiLU {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            return torch::silu(std::move(input));
        }
    };

}

#endif //THOT_SILU_HPP