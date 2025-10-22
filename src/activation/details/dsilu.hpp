#ifndef THOT_DSILU_HPP
#define THOT_DSILU_HPP

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Thot::Activation::Details {
    struct dSiLU {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            auto tensor = std::move(input);
            auto sigmoid = torch::sigmoid(tensor);
            return sigmoid * (1 + tensor * (1 - sigmoid));
        }
    };
}

#endif //THOT_DSILU_HPP