#ifndef Nott_PSILU_HPP
#define Nott_PSILU_HPP

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Nott::Activation::Details {

    struct PSiLU {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            constexpr double beta = 1.5;
            auto tensor = std::move(input);
            return tensor * torch::sigmoid(beta * tensor);
        }
    };

}
#endif //Nott_PSILU_HPP
