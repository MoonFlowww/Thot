#ifndef OMNI_PSILU_HPP
#define OMNI_PSILU_HPP

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Omni::Activation::Details {

    struct PSiLU {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            constexpr double beta = 1.5;
            auto tensor = std::move(input);
            return tensor * torch::sigmoid(beta * tensor);
        }
    };

}
#endif //OMNI_PSILU_HPP
