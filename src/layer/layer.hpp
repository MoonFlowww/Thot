#ifndef THOT_LAYER_HPP
#define THOT_LAYER_HPP
// This file is a factory, must exempt it from any logical-code. For functions look into "/details"
#include <variant>
#include <vector>
#include <utility>

#include "../common/local.hpp"
#include "details/batchnorm.hpp"
#include "details/conv.hpp"
#include "details/dropout.hpp"
#include "details/fc.hpp"
#include "details/flatten.hpp"
#include "details/pooling.hpp"
#include "registry.hpp"

namespace Thot::Layer {
    using FCOptions = Details::FCOptions;
    using FCDescriptor = Details::FCDescriptor;

    using Conv2dOptions = Details::Conv2dOptions;
    using Conv2dDescriptor = Details::Conv2dDescriptor;

    using BatchNorm2dOptions = Details::BatchNorm2dOptions;
    using BatchNorm2dDescriptor = Details::BatchNorm2dDescriptor;

    using MaxPool2dOptions = Details::MaxPool2dOptions;
    using AvgPool2dOptions = Details::AvgPool2dOptions;
    using AdaptiveAvgPool2dOptions = Details::AdaptiveAvgPool2dOptions;
    using AdaptiveMaxPool2dOptions = Details::AdaptiveMaxPool2dOptions;
    using PoolingDescriptor = Details::PoolingDescriptor;

    using HardDropoutOptions = Details::HardDropoutOptions;
    using HardDropoutDescriptor = Details::HardDropoutDescriptor;

    using FlattenOptions = Details::FlattenOptions;
    using FlattenDescriptor = Details::FlattenDescriptor;

    using Descriptor = std::variant<FCDescriptor,
                                    Conv2dDescriptor,
                                    BatchNorm2dDescriptor,
                                    PoolingDescriptor,
                                    HardDropoutDescriptor,
                                    FlattenDescriptor>;

    [[nodiscard]] inline auto FC(const FCOptions& options,
                                 ::Thot::Activation::Descriptor activation = ::Thot::Activation::Identity,
                                 ::Thot::Initialization::Descriptor initialization = ::Thot::Initialization::Default, ::Thot::LocalConfig local = {}) -> FCDescriptor {
        return {options, activation, initialization, std::move(local)};
    }

    [[nodiscard]] inline auto Conv2d(const Conv2dOptions& options, ::Thot::Activation::Descriptor activation = ::Thot::Activation::Identity,
                                     ::Thot::Initialization::Descriptor initialization = ::Thot::Initialization::Default, ::Thot::LocalConfig local = {}) -> Conv2dDescriptor {
        return {options, activation, initialization, std::move(local)};
    }

    [[nodiscard]] inline auto BatchNorm2d(const BatchNorm2dOptions& options,
                                              ::Thot::Activation::Descriptor activation = ::Thot::Activation::Identity,
                                              ::Thot::Initialization::Descriptor initialization = ::Thot::Initialization::Default, ::Thot::LocalConfig local = {})
            -> BatchNorm2dDescriptor
    {
        return {options, activation, initialization, std::move(local)};
    }

    [[nodiscard]] inline auto MaxPool2d(const MaxPool2dOptions& options,
                                        ::Thot::Activation::Descriptor activation = ::Thot::Activation::Identity,  ::Thot::LocalConfig local = {})
        -> PoolingDescriptor
    {
        PoolingDescriptor descriptor{};
        descriptor.options = options;
        descriptor.activation = activation;
        descriptor.local = std::move(local);
        return descriptor;
    }

    [[nodiscard]] inline auto AvgPool2d(const AvgPool2dOptions& options,
                                        ::Thot::Activation::Descriptor activation = ::Thot::Activation::Identity,  ::Thot::LocalConfig local = {})
        -> PoolingDescriptor
    {
        PoolingDescriptor descriptor{};
        descriptor.options = options;
        descriptor.activation = activation;
        descriptor.local = std::move(local);
        return descriptor;
    }

    [[nodiscard]] inline auto AdaptiveAvgPool2d(const AdaptiveAvgPool2dOptions& options,
                                                ::Thot::Activation::Descriptor activation = ::Thot::Activation::Identity,  ::Thot::LocalConfig local = {})
        -> PoolingDescriptor
    {
        PoolingDescriptor descriptor{};
        descriptor.options = options;
        descriptor.activation = activation;
        descriptor.local = std::move(local);
        return descriptor;
    }

    [[nodiscard]] inline auto AdaptiveMaxPool2d(const AdaptiveMaxPool2dOptions& options,
                                                ::Thot::Activation::Descriptor activation = ::Thot::Activation::Identity,  ::Thot::LocalConfig local = {})
        -> PoolingDescriptor
    {
        PoolingDescriptor descriptor{};
        descriptor.options = options;
        descriptor.activation = activation;
        descriptor.local = std::move(local);
        return descriptor;
    }

    [[nodiscard]] inline auto HardDropout(const HardDropoutOptions& options, ::Thot::Activation::Descriptor activation = ::Thot::Activation::Identity,  ::Thot::LocalConfig local = {}) -> HardDropoutDescriptor {
        return {options, activation, std::move(local)};
    }

    [[nodiscard]] inline auto Flatten(const FlattenOptions& options = {},
                                      ::Thot::Activation::Descriptor activation = ::Thot::Activation::Identity,  ::Thot::LocalConfig local = {})
        -> FlattenDescriptor
    {
        return {options, activation, std::move(local)};
    }

}

#endif //THOT_LAYER_HPP
