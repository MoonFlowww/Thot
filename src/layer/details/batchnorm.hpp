#ifndef THOT_BATCHNORM_HPP
#define THOT_BATCHNORM_HPP
#include <cstdint>

#include <stdexcept>
#include <string>
#include <utility>

#include <torch/nn/module.h>
#include <torch/nn/options/batchnorm.h>

#include "../../common/local.hpp"
#include "../../activation/activation.hpp"
#include "../../initialization/initialization.hpp"
#include "../../initialization/apply.hpp"
#include "../../initialization/initialization.hpp"
#include "../registry.hpp"

namespace Thot::Layer::Details {

    struct BatchNorm2dOptions {
        std::int64_t num_features{};
        double eps{1e-5};
        double momentum{0.1};
        bool affine{true};
        bool track_running_stats{true};
    };

    struct BatchNorm2dDescriptor {
        BatchNorm2dOptions options{};
        ::Thot::Activation::Descriptor activation{::Thot::Activation::Identity};
        ::Thot::Initialization::Descriptor initialization{::Thot::Initialization::Default};
        ::Thot::LocalConfig local{};
    };

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const BatchNorm2dDescriptor& descriptor, std::size_t index)
    {
        if (descriptor.options.num_features <= 0) {
            throw std::invalid_argument("BatchNorm2d requires a positive number of features.");
        }

        auto options = torch::nn::BatchNorm2dOptions(descriptor.options.num_features)
                            .eps(descriptor.options.eps)
                            .momentum(descriptor.options.momentum)
                            .affine(descriptor.options.affine)
                            .track_running_stats(descriptor.options.track_running_stats);

        auto module = owner.register_module("batchnorm2d_" + std::to_string(index), torch::nn::BatchNorm2d(options));
        ::Thot::Initialization::Details::apply_module_initialization(module, descriptor);

        RegisteredLayer registered_layer{};
        registered_layer.activation = descriptor.activation.type;
        registered_layer.module = to_shared_module_ptr(module);
        registered_layer.local = descriptor.local;
        registered_layer.forward = [module](torch::Tensor input) { return module->forward(std::move(input)); };
        return registered_layer;
    }

}

#endif //THOT_BATCHNORM_HPP