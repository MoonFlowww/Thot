#ifndef OMNI_ACTIVATION_APPLY_HPP
#define OMNI_ACTIVATION_APPLY_HPP

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

namespace Omni::Activation::Details {
    inline torch::Tensor apply(::Omni::Activation::Type type, torch::Tensor input) {
        switch (type) {
            case ::Omni::Activation::Type::ReLU:
                return ReLU{}(std::move(input));
            case ::Omni::Activation::Type::Sigmoid:
                return Sigmoid{}(std::move(input));
            case ::Omni::Activation::Type::Tanh:
                return Tanh{}(std::move(input));
            case ::Omni::Activation::Type::LeakyReLU:
                return LeakyReLU{}(std::move(input));
            case ::Omni::Activation::Type::Softmax:
                return Softmax{}(std::move(input));
            case ::Omni::Activation::Type::SiLU:
                return SiLU{}(std::move(input));
            case ::Omni::Activation::Type::GeLU:
                return GeLU{}(std::move(input));
            case ::Omni::Activation::Type::GLU:
                return GLU{}(std::move(input));
            case ::Omni::Activation::Type::SwiGLU:
                return SwiGLU{}(std::move(input));
            case ::Omni::Activation::Type::dSiLU:
                return dSiLU{}(std::move(input));
            case ::Omni::Activation::Type::PSiLU:
                return PSiLU{}(std::move(input));
            case ::Omni::Activation::Type::Mish:
                return Mish{}(std::move(input));
            case ::Omni::Activation::Type::Swish:
                return Swish{}(std::move(input));
            case ::Omni::Activation::Type::Identity:
                return input;
            default:
                return input;
        }
    }
}
#endif // OMNI_ACTIVATION_APPLY_HPP