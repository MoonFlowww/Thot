#ifndef Nott_DSILU_HPP
#define Nott_DSILU_HPP

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Nott::Activation::Details {
    struct dSiLU {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            auto tensor = std::move(input);
            auto sigmoid = torch::sigmoid(tensor);
            return sigmoid * (1 + tensor * (1 - sigmoid));
        }
    };
}

#endif //Nott_DSILU_HPP