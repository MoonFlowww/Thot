#ifndef OMNI_DSILU_HPP
#define OMNI_DSILU_HPP

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Omni::Activation::Details {
    struct dSiLU {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            auto tensor = std::move(input);
            auto sigmoid = torch::sigmoid(tensor);
            return sigmoid * (1 + tensor * (1 - sigmoid));
        }
    };
}

#endif //OMNI_DSILU_HPP