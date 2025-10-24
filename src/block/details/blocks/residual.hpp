#ifndef THOT_BLOCK_DETAILS_RESIDUAL_HPP
#define THOT_BLOCK_DETAILS_RESIDUAL_HPP

#include <cstddef>
#include <optional>
#include <vector>
#include <stdexcept>
#include <utility>

#include <torch/torch.h>

#include "../../../common/local.hpp"
#include "../../../activation/activation.hpp"
#include "../../../activation/apply.hpp"
#include "../../../layer/layer.hpp"

namespace Thot::Block::Details {
    struct ResidualSkipOptions {
        bool use_projection{false};
        std::optional<::Thot::Layer::Descriptor> projection{};
    };

    struct ResidualOutputOptions {
        ::Thot::Activation::Descriptor final_activation{::Thot::Activation::Identity};
        double dropout{0.0};
    };

    struct ResidualDescriptor {
        std::vector<::Thot::Layer::Descriptor> layers{};
        std::size_t repeats{1};
        ResidualSkipOptions skip{};
        ResidualOutputOptions output{};
        ::Thot::LocalConfig local{};
    };

     class ResidualBlockImpl : public torch::nn::Module {
    public:
        explicit ResidualBlockImpl(ResidualDescriptor descriptor)
            : final_activation_(descriptor.output.final_activation.type)
        {
            if (descriptor.layers.empty()) {
                throw std::invalid_argument("Residual blocks require at least one layer descriptor.");
            }
            if (descriptor.repeats == 0) {
                throw std::invalid_argument("Residual blocks require a positive repeat count.");
            }

            std::size_t module_index = 0;
            block_layers_.reserve(descriptor.repeats);
            for (std::size_t repeat = 0; repeat < descriptor.repeats; ++repeat) {
                std::vector<::Thot::Layer::Details::RegisteredLayer> registered_layers{};
                registered_layers.reserve(descriptor.layers.size());
                for (const auto& layer_descriptor : descriptor.layers) {
                    registered_layers.push_back(::Thot::Layer::Details::build_registered_layer(
                        *this, layer_descriptor, module_index++));
                }
                block_layers_.emplace_back(std::move(registered_layers));
            }

            const bool use_projection = descriptor.skip.use_projection || descriptor.skip.projection.has_value();
            if (use_projection) {
                if (!descriptor.skip.projection.has_value()) {
                    throw std::invalid_argument(
                        "Residual projection requested but no descriptor was provided.");
                }
                projection_layer_ = ::Thot::Layer::Details::build_registered_layer(
                    *this, *descriptor.skip.projection, module_index++);
            }

            if (descriptor.output.dropout > 0.0) {
                dropout_ = register_module(
                    "dropout",
                    torch::nn::Dropout(torch::nn::DropoutOptions(descriptor.output.dropout)));
            }
        }

        torch::Tensor forward(torch::Tensor input)
        {
            auto output = std::move(input);
            for (const auto& layers : block_layers_) {
                auto residual = output;
                auto branch = output;

                for (const auto& layer : layers) {
                    branch = layer.forward(std::move(branch));
                    branch = ::Thot::Activation::Details::apply(layer.activation, std::move(branch));
                }

                auto skip_connection = residual;
                if (projection_layer_.has_value()) {
                    skip_connection = projection_layer_->forward(std::move(skip_connection));
                    skip_connection = ::Thot::Activation::Details::apply(
                        projection_layer_->activation, std::move(skip_connection));
                }

                branch = std::move(branch) + std::move(skip_connection);
                branch = ::Thot::Activation::Details::apply(final_activation_, std::move(branch));
                if (dropout_) {
                    branch = dropout_->forward(branch);
                }

                output = std::move(branch);
            }

            return output;
        }

    private:
        std::vector<std::vector<::Thot::Layer::Details::RegisteredLayer>> block_layers_{};
        std::optional<::Thot::Layer::Details::RegisteredLayer> projection_layer_{};
        ::Thot::Activation::Type final_activation_{::Thot::Activation::Type::Identity};
        torch::nn::Dropout dropout_{nullptr};
    };

    TORCH_MODULE(ResidualBlock);
}

#endif // THOT_BLOCK_DETAILS_RESIDUAL_HPP