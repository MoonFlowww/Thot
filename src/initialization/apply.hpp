#ifndef THOT_INITIALIZATION_APPLY_HPP
#define THOT_INITIALIZATION_APPLY_HPP
#include <torch/torch.h>

#include "initialization.hpp"

namespace Thot::Initialization::Details {
    namespace detail {
        template <class Module>
        inline void zero_bias_if_present(const Module& module) {
            if constexpr (requires { module->bias; }) {
                if (module->bias.defined()) {
                    torch::nn::init::zeros_(module->bias);
                }
            }
        }
    }  // namespace detail

    template <class Module, class Descriptor>
    inline void apply_module_initialization(const Module& module, const Descriptor& descriptor) {
        const auto type = descriptor.initialization.type;

        switch (type) {
            case ::Thot::Initialization::Type::XavierNormal:
                torch::nn::init::xavier_normal_(module->weight);
                detail::zero_bias_if_present(module);
                break;
            case ::Thot::Initialization::Type::XavierUniform:
                torch::nn::init::xavier_uniform_(module->weight);
                detail::zero_bias_if_present(module);
                break;
            case ::Thot::Initialization::Type::KaimingNormal:
                torch::nn::init::kaiming_normal_(module->weight,
                                                 /*a=*/0.0,
                                                 torch::kFanIn,
                                                 torch::kReLU);
                detail::zero_bias_if_present(module);
                break;
            case ::Thot::Initialization::Type::KaimingUniform:
                torch::nn::init::kaiming_uniform_(module->weight,
                                                  /*a=*/0.0,
                                                  torch::kFanIn,
                                                  torch::kReLU);
                detail::zero_bias_if_present(module);
                break;
            case ::Thot::Initialization::Type::Dirac:
                if (module->weight.dim() >= 3) {
                    torch::nn::init::dirac_(module->weight);
                }
                detail::zero_bias_if_present(module);
                break;
            case ::Thot::Initialization::Type::Lyapunov:
                torch::nn::init::orthogonal_(module->weight);
                detail::zero_bias_if_present(module);
                break;
            case ::Thot::Initialization::Type::ZeroBias:
                detail::zero_bias_if_present(module);
                break;
            case ::Thot::Initialization::Type::Default:
            default:
                break;
        }
    }
}
#endif // THOT_INITIALIZATION_APPLY_HPP