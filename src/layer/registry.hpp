#ifndef THOT_LAYER_REGISTRY_HPP
#define THOT_LAYER_REGISTRY_HPP

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

#include <torch/torch.h>

#include "../activation/activation.hpp"
#include "../initialization/apply.hpp"
#include "details/fc.hpp"

namespace Thot::Layer::Details {
    struct RegisteredLayer {
        std::function<torch::Tensor(torch::Tensor)> forward{};
        ::Thot::Activation::Type activation{::Thot::Activation::Type::Identity};
    };

    template <class Owner, class Descriptor>
    RegisteredLayer build_registered_layer(Owner&, const Descriptor&, std::size_t) {
        static_assert(sizeof(Descriptor) == 0, "Unsupported layer descriptor provided to build_registered_layer.");
        return {};
    }

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const FCDescriptor& descriptor, std::size_t index) {
        if (descriptor.options.in_features <= 0 || descriptor.options.out_features <= 0) {
            throw std::invalid_argument("Fully connected layers require positive in/out features.");
        }

        auto options = torch::nn::LinearOptions(descriptor.options.in_features, descriptor.options.out_features)
                            .bias(descriptor.options.bias);
        auto module = owner.register_module("fc_" + std::to_string(index), torch::nn::Linear(options));
        ::Thot::Initialization::Details::apply_module_initialization(module, descriptor);

        RegisteredLayer registered_layer{};
        registered_layer.activation = descriptor.activation.type;
        registered_layer.forward = [module](torch::Tensor input) {
            return module->forward(std::move(input));
        };
        return registered_layer;
    }
}

#endif // THOT_LAYER_REGISTRY_HPP
