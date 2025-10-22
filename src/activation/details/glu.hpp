#ifndef THOT_GLU_HPP
#define THOT_GLU_HPP


#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Thot::Activation::Details {

    struct GLU {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            if (input.dim() == 0) {
                return input;
            }
            const auto dim = input.dim() - 1;
            return torch::glu(std::move(input), dim);
        }
    };

}

#endif //THOT_GLU_HPP