#ifndef THOT_RELU_HPP
#define THOT_RELU_HPP
#include <torch/torch.h>
#include <torch/nn/functional/functional.h>

namespace Thot::Activation::Details {
    struct ReLU {
        struct Tag {};

        static constexpr Tag tag{};

        [[nodiscard]] static inline torch::Tensor apply(const torch::Tensor &input) {
            return torch::relu(input);
        }

        [[nodiscard]] static inline torch::Tensor derivative(const torch::Tensor &input) {
            auto grad_output = torch::ones_like(input);
            return torch::nn::functional::relu_backward(grad_output, input);
        }
    };

    constexpr ReLU::Tag ReLU::tag;
}

#endif //THOT_RELU_HPP