#ifndef THOT_BLOCK_HPP
#define THOT_BLOCK_HPP
// This file is an factory, must exempt it from any logical-code. For functions look into "/details"
#include <cstddef>
#include <initializer_list>
#include <utility>
#include <variant>
#include <vector>

#include "details/residual.hpp"
#include "details/sequential.hpp"
#include "details/transformers/classic.hpp"

namespace Thot::Block {
    using SequentialDescriptor = Details::SequentialDescriptor;
    using ResidualDescriptor = Details::ResidualDescriptor;

    namespace Transformer = Details::Transformer;

    using Descriptor = std::variant<SequentialDescriptor,
                                    ResidualDescriptor,
                                    Transformer::Classic::EncoderDescriptor,
                                    Transformer::Classic::DecoderDescriptor>;

    [[nodiscard]] inline auto Sequential(std::initializer_list<::Thot::Layer::Descriptor> layers)
        -> SequentialDescriptor
    {
        return SequentialDescriptor{std::vector<::Thot::Layer::Descriptor>(layers.begin(), layers.end())};
    }

    [[nodiscard]] inline auto Sequential(std::vector<::Thot::Layer::Descriptor> layers)
        -> SequentialDescriptor
    {
        return SequentialDescriptor{std::move(layers)};
    }

    [[nodiscard]] inline auto Residual(std::initializer_list<::Thot::Layer::Descriptor> layers,
                                       std::size_t repeats = 1,
                                       Details::ResidualSkipOptions skip = {},
                                       Details::ResidualOutputOptions output = {})
        -> ResidualDescriptor
    {
        ResidualDescriptor descriptor{};
        descriptor.layers.assign(layers.begin(), layers.end());
        descriptor.repeats = repeats;
        descriptor.skip = std::move(skip);
        descriptor.output = std::move(output);
        return descriptor;
    }

    [[nodiscard]] inline auto Residual(std::vector<::Thot::Layer::Descriptor> layers,
                                       std::size_t repeats = 1,
                                       Details::ResidualSkipOptions skip = {},
                                       Details::ResidualOutputOptions output = {})
        -> ResidualDescriptor
    {
        ResidualDescriptor descriptor{};
        descriptor.layers = std::move(layers);
        descriptor.repeats = repeats;
        descriptor.skip = std::move(skip);
        descriptor.output = std::move(output);
        return descriptor;
    }
}

#endif //THOT_BLOCK_HPP