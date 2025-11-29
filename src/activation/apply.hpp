#ifndef Nott_ACTIVATION_APPLY_HPP
#define Nott_ACTIVATION_APPLY_HPP

#include <torch/torch.h>

#include <utility>

#include "activation.hpp"
#include "details/dsilu.hpp"
#include "details/gelu.hpp"
#include "details/glu.hpp"
#include "details/leaky_relu.hpp"
#include "details/mish.hpp"
#include "details/psilu.hpp"
#include "details/relu.hpp"
#include "details/sigmoid.hpp"
#include "details/silu.hpp"
#include "details/softmax.hpp"
#include "details/swiglu.hpp"
#include "details/swish.hpp"
#include "details/tanh.hpp"

namespace Nott::Activation::Details {
    inline torch::Tensor apply(::Nott::Activation::Type type, torch::Tensor input) {
        switch (type) {
            case ::Nott::Activation::Type::ReLU:
                return ReLU{}(std::move(input));
            case ::Nott::Activation::Type::Sigmoid:
                return Sigmoid{}(std::move(input));
            case ::Nott::Activation::Type::Tanh:
                return Tanh{}(std::move(input));
            case ::Nott::Activation::Type::LeakyReLU:
                return LeakyReLU{}(std::move(input));
            case ::Nott::Activation::Type::Softmax:
                return Softmax{}(std::move(input));
            case ::Nott::Activation::Type::SiLU:
                return SiLU{}(std::move(input));
            case ::Nott::Activation::Type::GeLU:
                return GeLU{}(std::move(input));
            case ::Nott::Activation::Type::GLU:
                return GLU{}(std::move(input));
            case ::Nott::Activation::Type::SwiGLU:
                return SwiGLU{}(std::move(input));
            case ::Nott::Activation::Type::dSiLU:
                return dSiLU{}(std::move(input));
            case ::Nott::Activation::Type::PSiLU:
                return PSiLU{}(std::move(input));
            case ::Nott::Activation::Type::Mish:
                return Mish{}(std::move(input));
            case ::Nott::Activation::Type::Swish:
                return Swish{}(std::move(input));
            case ::Nott::Activation::Type::Identity:
                return input;
            default:
                return input;
        }
    }
}
#endif // Nott_ACTIVATION_APPLY_HPP