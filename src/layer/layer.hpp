#ifndef THOT_LAYER_HPP
#define THOT_LAYER_HPP
// This file is a factory, must exempt it from any logical-code. For functions look into "/details"
#include <variant>
#include <vector>

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

    using DropoutOptions = Details::DropoutOptions;
    using DropoutDescriptor = Details::DropoutDescriptor;

    using FlattenOptions = Details::FlattenOptions;
    using FlattenDescriptor = Details::FlattenDescriptor;

    using Descriptor = std::variant<FCDescriptor,
                                    Conv2dDescriptor,
                                    BatchNorm2dDescriptor,
                                    PoolingDescriptor,
                                    DropoutDescriptor,
                                    FlattenDescriptor>;

    [[nodiscard]] inline auto FC(const FCOptions& options,
                                 ::Thot::Activation::Descriptor activation = ::Thot::Activation::Identity,
                                 ::Thot::Initialization::Descriptor initialization = ::Thot::Initialization::Default)
        -> FCDescriptor
    {
        return {options, activation, initialization};
    }

    [[nodiscard]] inline auto Conv2d(const Conv2dOptions& options,
                                     ::Thot::Activation::Descriptor activation = ::Thot::Activation::Identity,
                                     ::Thot::Initialization::Descriptor initialization = ::Thot::Initialization::Default)
        -> Conv2dDescriptor
    {
        return {options, activation, initialization};
    }

    [[nodiscard]] inline auto BatchNorm2d(const BatchNorm2dOptions& options,
                                              ::Thot::Activation::Descriptor activation = ::Thot::Activation::Identity,
                                              ::Thot::Initialization::Descriptor initialization = ::Thot::Initialization::Default)
            -> BatchNorm2dDescriptor
    {
        return {options, activation, initialization};
    }

    [[nodiscard]] inline auto MaxPool2d(const MaxPool2dOptions& options,
                                        ::Thot::Activation::Descriptor activation = ::Thot::Activation::Identity)
        -> PoolingDescriptor
    {
        PoolingDescriptor descriptor{};
        descriptor.options = options;
        descriptor.activation = activation;
        return descriptor;
    }

    [[nodiscard]] inline auto AvgPool2d(const AvgPool2dOptions& options,
                                        ::Thot::Activation::Descriptor activation = ::Thot::Activation::Identity)
        -> PoolingDescriptor
    {
        PoolingDescriptor descriptor{};
        descriptor.options = options;
        descriptor.activation = activation;
        return descriptor;
    }

    [[nodiscard]] inline auto AdaptiveAvgPool2d(const AdaptiveAvgPool2dOptions& options,
                                                ::Thot::Activation::Descriptor activation = ::Thot::Activation::Identity)
        -> PoolingDescriptor
    {
        PoolingDescriptor descriptor{};
        descriptor.options = options;
        descriptor.activation = activation;
        return descriptor;
    }

    [[nodiscard]] inline auto AdaptiveMaxPool2d(const AdaptiveMaxPool2dOptions& options,
                                                ::Thot::Activation::Descriptor activation = ::Thot::Activation::Identity)
        -> PoolingDescriptor
    {
        PoolingDescriptor descriptor{};
        descriptor.options = options;
        descriptor.activation = activation;
        return descriptor;
    }

    [[nodiscard]] inline auto Dropout(const DropoutOptions& options,
                                      ::Thot::Activation::Descriptor activation = ::Thot::Activation::Identity)
        -> DropoutDescriptor
    {
        return {options, activation};
    }

    [[nodiscard]] inline auto Flatten(const FlattenOptions& options = {},
                                      ::Thot::Activation::Descriptor activation = ::Thot::Activation::Identity)
        -> FlattenDescriptor
    {
        return {options, activation};
    }

}

#endif //THOT_LAYER_HPP
