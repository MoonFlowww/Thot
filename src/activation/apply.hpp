#ifndef THOT_APPLY_HPP
#define THOT_APPLY_HPP

#include <torch/torch.h>

#include <utility>

#include "activation.hpp"
#include "details/dsilu.hpp"
#include "details/gelu.hpp"
#include "details/glu.hpp"
#include "details/leaky_relu.hpp"
#include "details/mish.hpp"
#include "details/psilu.hpp"
#include "details/raw.hpp"
#include "details/relu.hpp"
#include "details/sigmoid.hpp"
#include "details/silu.hpp"
#include "details/softmax.hpp"
#include "details/swiglu.hpp"
#include "details/swish.hpp"
#include "details/tanh.hpp"

namespace Thot::Activation::Details {
    inline torch::Tensor apply(::Thot::Activation::Type type, torch::Tensor input) {
        switch (type) {
            case ::Thot::Activation::Type::Raw:
                return Raw{}(std::move(input));
            case ::Thot::Activation::Type::ReLU:
                return ReLU{}(std::move(input));
            case ::Thot::Activation::Type::Sigmoid:
                return Sigmoid{}(std::move(input));
            case ::Thot::Activation::Type::Tanh:
                return Tanh{}(std::move(input));
            case ::Thot::Activation::Type::LeakyReLU:
                return LeakyReLU{}(std::move(input));
            case ::Thot::Activation::Type::Softmax:
                return Softmax{}(std::move(input));
            case ::Thot::Activation::Type::SiLU:
                return SiLU{}(std::move(input));
            case ::Thot::Activation::Type::GeLU:
                return GeLU{}(std::move(input));
            case ::Thot::Activation::Type::GLU:
                return GLU{}(std::move(input));
            case ::Thot::Activation::Type::SwiGLU:
                return SwiGLU{}(std::move(input));
            case ::Thot::Activation::Type::dSiLU:
                return dSiLU{}(std::move(input));
            case ::Thot::Activation::Type::PSiLU:
                return PSiLU{}(std::move(input));
            case ::Thot::Activation::Type::Mish:
                return Mish{}(std::move(input));
            case ::Thot::Activation::Type::Swish:
                return Swish{}(std::move(input));
            case ::Thot::Activation::Type::Identity:
            default:
                return input;
        }
    }
}
#endif //THOT_APPLY_HPP