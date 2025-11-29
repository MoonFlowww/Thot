#ifndef Nott_FC_HPP
#define Nott_FC_HPP

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include <torch/nn/module.h>
#include <torch/nn/options/linear.h>
#include "../../activation/activation.hpp"
#include "../../common/local.hpp"
#include "../../initialization/apply.hpp"
#include "../../initialization/initialization.hpp"
#include "../registry.hpp"


namespace Nott::Layer::Details {
    struct FCOptions {
        std::int64_t in_features{};
        std::int64_t out_features{};
        bool bias{true};
    };

    struct FCDescriptor {
        FCOptions options;
        ::Nott::Activation::Descriptor activation{::Nott::Activation::Identity};
        ::Nott::Initialization::Descriptor initialization{::Nott::Initialization::Default};
        ::Nott::LocalConfig local{};
    };
    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const FCDescriptor& descriptor, std::size_t index)
    {
        if (descriptor.options.in_features <= 0 || descriptor.options.out_features <= 0) {
            throw std::invalid_argument("Fully connected layers require positive in/out features.");
        }

        auto options = torch::nn::LinearOptions(descriptor.options.in_features, descriptor.options.out_features)
                            .bias(descriptor.options.bias);
        auto module = owner.register_module("fc_" + std::to_string(index), torch::nn::Linear(options));
        ::Nott::Initialization::Details::apply_module_initialization(module, descriptor);

        RegisteredLayer registered_layer{};
        registered_layer.activation = descriptor.activation.type;
        registered_layer.module = to_shared_module_ptr(module);
        registered_layer.local = descriptor.local;
        registered_layer.bind_module_forward(module.get());
        return registered_layer;
    }
}

#endif //Nott_FC_HPP
