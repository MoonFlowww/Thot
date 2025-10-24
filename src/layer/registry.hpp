#ifndef THOT_LAYER_REGISTRY_HPP
#define THOT_LAYER_REGISTRY_HPP

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include <torch/torch.h>

#include "../activation/activation.hpp"
#include "../common/local.hpp"
#include "../initialization/apply.hpp"
#include "details/batchnorm.hpp"
#include "details/conv.hpp"
#include "details/dropout.hpp"
#include "details/fc.hpp"
#include "details/flatten.hpp"
#include "details/pooling.hpp"

namespace Thot::Layer::Details {

    namespace detail {
        template <class Descriptor>
        struct is_supported_descriptor : std::false_type {};

        template <class Descriptor>
        using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<Descriptor>>;

        template <class Descriptor>
        inline constexpr bool dependent_false_v = false;

        template <class Descriptor>
        inline constexpr bool is_supported_descriptor_v =
            is_supported_descriptor<remove_cvref_t<Descriptor>>::value;

        template <class Descriptor>
        struct is_variant : std::false_type {};
        template <class... Types>
        struct is_variant<std::variant<Types...>> : std::true_type {};
        template <class Descriptor>
        inline constexpr bool is_variant_v = is_variant<remove_cvref_t<Descriptor>>::value;

        template <>
        struct is_supported_descriptor<FCDescriptor> : std::true_type {};
        template <>
        struct is_supported_descriptor<Conv2dDescriptor> : std::true_type {};
        template <>
        struct is_supported_descriptor<BatchNorm2dDescriptor> : std::true_type {};
        template <>
        struct is_supported_descriptor<PoolingDescriptor> : std::true_type {};
        template <>
        struct is_supported_descriptor<DropoutDescriptor> : std::true_type {};
        template <>
        struct is_supported_descriptor<FlattenDescriptor> : std::true_type {};
    }

    struct RegisteredLayer {
        std::function<torch::Tensor(torch::Tensor)> forward{};
        ::Thot::Activation::Type activation{::Thot::Activation::Type::Identity};
        std::shared_ptr<torch::nn::Module> module{};
        ::Thot::LocalConfig local{};
    };

    template <class Owner, class Descriptor, std::enable_if_t<!detail::is_supported_descriptor_v<Descriptor> && !detail::is_variant_v<Descriptor>, int> = 0>
    RegisteredLayer build_registered_layer(Owner&, const Descriptor&, std::size_t) {
        static_assert(detail::dependent_false_v<Descriptor>, "Unsupported layer descriptor provided to build_registered_layer.");
        return {};
    }


    template <class Owner, class... DescriptorTypes>
    RegisteredLayer build_registered_layer(Owner& owner,
                                           const std::variant<DescriptorTypes...>& descriptor,
                                           std::size_t index)
    {
        return std::visit(
            [&](const auto& concrete_descriptor) {
                return build_registered_layer(owner, concrete_descriptor, index);
            },
            descriptor);
    }


    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const FCDescriptor& descriptor, std::size_t index) {
        if (descriptor.options.in_features <= 0 || descriptor.options.out_features <= 0) {
            throw std::invalid_argument("Fully connected layers require positive in/out features.");
        }

        auto options = torch::nn::LinearOptions(descriptor.options.in_features, descriptor.options.out_features)
                            .bias(descriptor.options.bias);
        auto module = owner.register_module("fc_" + std::to_string(index), torch::nn::Linear(options));
        auto module_ptr = module.ptr();
        ::Thot::Initialization::Details::apply_module_initialization(module, descriptor);

        RegisteredLayer registered_layer{};
        registered_layer.activation = descriptor.activation.type;
        registered_layer.module = std::static_pointer_cast<torch::nn::Module>(module_ptr);

        registered_layer.local = descriptor.local;
        registered_layer.forward = [module_ptr](torch::Tensor input) {
            return module_ptr->forward(std::move(input));
        };
        return registered_layer;
    }

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const Conv2dDescriptor& descriptor, std::size_t index)
    {
        if (descriptor.options.in_channels <= 0 || descriptor.options.out_channels <= 0) {
            throw std::invalid_argument("Conv2d layers require positive channel counts.");
        }

        auto options = torch::nn::Conv2dOptions(descriptor.options.in_channels,
                                                descriptor.options.out_channels,
                                                descriptor.options.kernel_size);

        if (!descriptor.options.stride.empty()) {
            options.stride(descriptor.options.stride);
        }
        if (!descriptor.options.padding.empty()) {
            options.padding(descriptor.options.padding);
        }
        if (!descriptor.options.dilation.empty()) {
            options.dilation(descriptor.options.dilation);
        }
        options.groups(descriptor.options.groups);
        options.bias(descriptor.options.bias);
        if (!descriptor.options.padding_mode.empty()) {
            auto padding_mode = descriptor.options.padding_mode;
            std::transform(padding_mode.begin(), padding_mode.end(), padding_mode.begin(),
                           [](unsigned char character) { return static_cast<char>(std::tolower(character)); });

            if (padding_mode == "zeros") {
                options.padding_mode(torch::kZeros);
            } else if (padding_mode == "reflect") {
                options.padding_mode(torch::kReflect);
            } else if (padding_mode == "replicate") {
                options.padding_mode(torch::kReplicate);
            } else if (padding_mode == "circular") {
                options.padding_mode(torch::kCircular);
            } else {
                throw std::invalid_argument("Unsupported padding mode provided to Conv2d descriptor: " +
                                            descriptor.options.padding_mode);
            }
        }

        auto module = owner.register_module("conv2d_" + std::to_string(index), torch::nn::Conv2d(options));
        ::Thot::Initialization::Details::apply_module_initialization(module, descriptor);

        RegisteredLayer registered_layer{};
        registered_layer.activation = descriptor.activation.type;
        registered_layer.module = std::static_pointer_cast<torch::nn::Module>(module.ptr());
        registered_layer.local = descriptor.local;
        registered_layer.forward = [module](torch::Tensor input) {
            return module->forward(std::move(input));
        };
        return registered_layer;
    }

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
        registered_layer.module = std::static_pointer_cast<torch::nn::Module>(module.ptr());
        registered_layer.local = descriptor.local;
        registered_layer.forward = [module](torch::Tensor input) {
            return module->forward(std::move(input));
        };
        return registered_layer;
    }

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const PoolingDescriptor& descriptor, std::size_t index)
    {
        RegisteredLayer registered_layer{};
        registered_layer.activation = descriptor.activation.type;
        registered_layer.local = descriptor.local;

        std::visit(
            [&](const auto& options) {
                using OptionType = std::decay_t<decltype(options)>;

                if constexpr (std::is_same_v<OptionType, MaxPool2dOptions>) {
                    auto torch_options = torch::nn::MaxPool2dOptions(options.kernel_size)
                                              .ceil_mode(options.ceil_mode);
                    if (!options.stride.empty()) {
                        torch_options.stride(options.stride);
                    }
                    if (!options.padding.empty()) {
                        torch_options.padding(options.padding);
                    }
                    if (!options.dilation.empty()) {
                        torch_options.dilation(options.dilation);
                    }

                    auto module = owner.register_module("maxpool2d_" + std::to_string(index), torch::nn::MaxPool2d(torch_options));
                    registered_layer.module = std::static_pointer_cast<torch::nn::Module>(module.ptr());
                    registered_layer.forward = [module](torch::Tensor input) {
                        return module->forward(std::move(input));
                    };
                } else if constexpr (std::is_same_v<OptionType, AvgPool2dOptions>) {
                    auto torch_options = torch::nn::AvgPool2dOptions(options.kernel_size)
                                              .ceil_mode(options.ceil_mode)
                                              .count_include_pad(options.count_include_pad);
                    if (!options.stride.empty()) {
                        torch_options.stride(options.stride);
                    }
                    if (!options.padding.empty()) {
                        torch_options.padding(options.padding);
                    }

                    auto module = owner.register_module("avgpool2d_" + std::to_string(index), torch::nn::AvgPool2d(torch_options));
                    registered_layer.module = std::static_pointer_cast<torch::nn::Module>(module.ptr());
                    registered_layer.forward = [module](torch::Tensor input) {
                        return module->forward(std::move(input));
                    };
                } else if constexpr (std::is_same_v<OptionType, AdaptiveAvgPool2dOptions>) {
                    auto module = owner.register_module("adaptive_avgpool2d_" + std::to_string(index),
                                                        torch::nn::AdaptiveAvgPool2d(options.output_size));
                    registered_layer.module = std::static_pointer_cast<torch::nn::Module>(module.ptr());
                    registered_layer.forward = [module](torch::Tensor input) {
                        return module->forward(std::move(input));
                    };
                } else if constexpr (std::is_same_v<OptionType, AdaptiveMaxPool2dOptions>) {
                    auto module = owner.register_module("adaptive_maxpool2d_" + std::to_string(index),
                                                        torch::nn::AdaptiveMaxPool2d(options.output_size));
                    registered_layer.module = std::static_pointer_cast<torch::nn::Module>(module.ptr());
                    registered_layer.forward = [module](torch::Tensor input) {
                        return module->forward(std::move(input));
                    };
                }
            },
            descriptor.options);

        if (!registered_layer.forward) {
            throw std::invalid_argument("Unsupported pooling descriptor provided.");
        }

        return registered_layer;
    }

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const DropoutDescriptor& descriptor, std::size_t index)
    {
        auto options = torch::nn::DropoutOptions(descriptor.options.probability).inplace(descriptor.options.inplace);
        auto module = owner.register_module("dropout_" + std::to_string(index), torch::nn::Dropout(options));

        RegisteredLayer registered_layer{};
        registered_layer.activation = descriptor.activation.type;
        registered_layer.module = std::static_pointer_cast<torch::nn::Module>(module.ptr());
        registered_layer.local = descriptor.local;
        registered_layer.forward = [module](torch::Tensor input) {
            return module->forward(std::move(input));
        };
        return registered_layer;
    }

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const FlattenDescriptor& descriptor, std::size_t index)
    {
        auto options = torch::nn::FlattenOptions().start_dim(descriptor.options.start_dim).end_dim(descriptor.options.end_dim);
        auto module = owner.register_module("flatten_" + std::to_string(index), torch::nn::Flatten(options));

        RegisteredLayer registered_layer{};
        registered_layer.activation = descriptor.activation.type;
        registered_layer.module = std::static_pointer_cast<torch::nn::Module>(module.ptr());
        registered_layer.local = descriptor.local;
        registered_layer.forward = [module](torch::Tensor input) {
            return module->forward(std::move(input));
        };
        return registered_layer;
    }

}

#endif // THOT_LAYER_REGISTRY_HPP
