#ifndef THOT_RAW_HPP
#define THOT_RAW_HPP

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Thot::Activation::Details {

    struct Raw {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            return input;
        }
    };

}

#endif //THOT_RAW_HPP