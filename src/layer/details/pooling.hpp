#ifndef OMNI_POOLING_HPP
#define OMNI_POOLING_HPP
#include <cstdint>
#include <variant>
#include <vector>
#include <stdexcept>
#include <string>
#include <utility>
#include <type_traits>

#include "../../activation/activation.hpp"
#include "../../common/local.hpp"
#include "../registry.hpp"

namespace Omni::Layer::Details {

    struct MaxPool1dOptions {
        std::vector<std::int64_t> kernel_size{2};
        std::vector<std::int64_t> stride{};
        std::vector<std::int64_t> padding{0};
        std::vector<std::int64_t> dilation{1};
        bool ceil_mode{false};
    };

    struct AvgPool1dOptions {
        std::vector<std::int64_t> kernel_size{2};
        std::vector<std::int64_t> stride{};
        std::vector<std::int64_t> padding{0};
        bool ceil_mode{false};
        bool count_include_pad{false};
    };

    struct AdaptiveAvgPool1dOptions {
        std::vector<std::int64_t> output_size{1};
    };

    struct AdaptiveMaxPool1dOptions {
        std::vector<std::int64_t> output_size{1};
    };

    struct MaxPool2dOptions {
        std::vector<std::int64_t> kernel_size{2, 2};
        std::vector<std::int64_t> stride{};
        std::vector<std::int64_t> padding{0, 0};
        std::vector<std::int64_t> dilation{1, 1};
        bool ceil_mode{false};
    };

    struct AvgPool2dOptions {
        std::vector<std::int64_t> kernel_size{2, 2};
        std::vector<std::int64_t> stride{};
        std::vector<std::int64_t> padding{0, 0};
        bool ceil_mode{false};
        bool count_include_pad{false};
    };

    struct AdaptiveAvgPool2dOptions {
        std::vector<std::int64_t> output_size{1, 1};
    };

    struct AdaptiveMaxPool2dOptions {
        std::vector<std::int64_t> output_size{1, 1};
    };

    using PoolingOptions = std::variant<MaxPool1dOptions,
                                    AvgPool1dOptions,
                                    AdaptiveAvgPool1dOptions,
                                    AdaptiveMaxPool1dOptions,
                                    MaxPool2dOptions,
                                    AvgPool2dOptions,
                                    AdaptiveAvgPool2dOptions,
                                    AdaptiveMaxPool2dOptions>;

    struct PoolingDescriptor {
        PoolingOptions options{};
        ::Omni::Activation::Descriptor activation{::Omni::Activation::Identity};
        ::Omni::LocalConfig local{};
    };

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const PoolingDescriptor& descriptor, std::size_t index)
    {
        RegisteredLayer registered_layer{};
        registered_layer.activation = descriptor.activation.type;
        registered_layer.local = descriptor.local;

        std::visit(
            [&](const auto& options) {
                using OptionType = std::decay_t<decltype(options)>;

                if constexpr (std::is_same_v<OptionType, MaxPool1dOptions>) {
                    auto torch_options = torch::nn::MaxPool1dOptions(options.kernel_size).ceil_mode(options.ceil_mode);
                    if (!options.stride.empty()) {
                        torch_options.stride(options.stride);
                    }
                    if (!options.padding.empty()) {
                        torch_options.padding(options.padding);
                    }
                    if (!options.dilation.empty()) {
                        torch_options.dilation(options.dilation);
                    }

                    auto module = owner.register_module("maxpool1d_" + std::to_string(index),
                                                        torch::nn::MaxPool1d(torch_options));
                    registered_layer.module = to_shared_module_ptr(module);
                    registered_layer.bind_module_forward(module.get());
                } else if constexpr (std::is_same_v<OptionType, AvgPool1dOptions>) {
                    auto torch_options = torch::nn::AvgPool1dOptions(options.kernel_size)
                                              .ceil_mode(options.ceil_mode)
                                              .count_include_pad(options.count_include_pad);
                    if (!options.stride.empty()) {
                        torch_options.stride(options.stride);
                    }
                    if (!options.padding.empty()) {
                        torch_options.padding(options.padding);
                    }

                    auto module = owner.register_module("avgpool1d_" + std::to_string(index),
                                                        torch::nn::AvgPool1d(torch_options));
                    registered_layer.module = to_shared_module_ptr(module);
                    registered_layer.bind_module_forward(module.get());
                } else if constexpr (std::is_same_v<OptionType, AdaptiveAvgPool1dOptions>) {
                    auto module = owner.register_module("adaptive_avgpool1d_" + std::to_string(index),
                                                        torch::nn::AdaptiveAvgPool1d(options.output_size));
                    registered_layer.module = to_shared_module_ptr(module);
                    registered_layer.bind_module_forward(module.get());
                } else if constexpr (std::is_same_v<OptionType, AdaptiveMaxPool1dOptions>) {
                    auto module = owner.register_module("adaptive_maxpool1d_" + std::to_string(index),
                                                        torch::nn::AdaptiveMaxPool1d(options.output_size));
                    registered_layer.module = to_shared_module_ptr(module);
                    registered_layer.bind_module_forward(module.get());
                } else if constexpr (std::is_same_v<OptionType, MaxPool2dOptions>) {
                    auto torch_options = torch::nn::MaxPool2dOptions(options.kernel_size).ceil_mode(options.ceil_mode);
                    if (!options.stride.empty()) {
                        torch_options.stride(options.stride);
                    }
                    if (!options.padding.empty()) {
                        torch_options.padding(options.padding);
                    }
                    if (!options.dilation.empty()) {
                        torch_options.dilation(options.dilation);
                    }

                    auto module = owner.register_module("maxpool2d_" + std::to_string(index),
                                                        torch::nn::MaxPool2d(torch_options));
                    registered_layer.module = to_shared_module_ptr(module);
                    registered_layer.bind_module_forward(module.get());
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

                    auto module = owner.register_module("avgpool2d_" + std::to_string(index),
                                                        torch::nn::AvgPool2d(torch_options));
                    registered_layer.module = to_shared_module_ptr(module);
                    registered_layer.bind_module_forward(module.get());
                } else if constexpr (std::is_same_v<OptionType, AdaptiveAvgPool2dOptions>) {
                    auto module = owner.register_module("adaptive_avgpool2d_" + std::to_string(index),
                                                        torch::nn::AdaptiveAvgPool2d(options.output_size));
                    registered_layer.module = to_shared_module_ptr(module);
                    registered_layer.bind_module_forward(module.get());
                } else if constexpr (std::is_same_v<OptionType, AdaptiveMaxPool2dOptions>) {
                    auto module = owner.register_module("adaptive_maxpool2d_" + std::to_string(index),
                                                        torch::nn::AdaptiveMaxPool2d(options.output_size));
                    registered_layer.module = to_shared_module_ptr(module);
                    registered_layer.bind_module_forward(module.get());
                }
            },
            descriptor.options);

        if (!registered_layer.forward) {
            throw std::invalid_argument("Unsupported pooling descriptor provided.");
        }

        return registered_layer;
    }

}

#endif //OMNI_POOLING_HPP