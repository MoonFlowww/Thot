#ifndef OMNI_BLOCK_DETAILS_RESIDUAL_HPP
#define OMNI_BLOCK_DETAILS_RESIDUAL_HPP

#include <cstddef>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <torch/torch.h>

#include "../../../common/local.hpp"
#include "../../../activation/activation.hpp"
#include "../../../activation/apply.hpp"
#include "../../../layer/layer.hpp"

namespace Omni::Block::Details {
    struct ResidualSkipOptions {
        std::optional<::Omni::Layer::Descriptor> projection{};
    };

    struct ResidualOutputOptions {
        ::Omni::Activation::Descriptor final_activation{::Omni::Activation::Identity};
        double dropout{0.0};
    };

    struct ResidualDescriptor {
        std::vector<::Omni::Layer::Descriptor> layers{};
        std::size_t repeats{1};
        ResidualSkipOptions skip{};
        ResidualOutputOptions output{};
        ::Omni::LocalConfig local{};
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
                std::vector<::Omni::Layer::Details::RegisteredLayer> registered_layers{};
                registered_layers.reserve(descriptor.layers.size());
                for (const auto& layer_descriptor : descriptor.layers) {
                    registered_layers.push_back(::Omni::Layer::Details::build_registered_layer(
                        *this, layer_descriptor, module_index++));
                }
                block_layers_.emplace_back(std::move(registered_layers));
            }

            if (descriptor.skip.projection.has_value()) {
                projection_layer_ = ::Omni::Layer::Details::build_registered_layer(
                    *this, *descriptor.skip.projection, module_index++);
            }

            if (descriptor.output.dropout > 0.0) {
                dropout_ = register_module(
                    "dropout",
                    torch::nn::Dropout(torch::nn::DropoutOptions(descriptor.output.dropout)));
            }
            apply_tensor_memory_format_to_convolutions();
        }

        void set_preferred_tensor_memory_format(torch::MemoryFormat format)
        {
            if (preferred_tensor_memory_format_ == format) {
                return;
            }

            preferred_tensor_memory_format_ = format;
            apply_tensor_memory_format_to_convolutions();
        }

        [[nodiscard]] torch::MemoryFormat preferred_tensor_memory_format() const noexcept
        {
            return preferred_tensor_memory_format_;
        }

        torch::Tensor forward(torch::Tensor input)
        {
            auto output = std::move(input);
            for (const auto& layers : block_layers_) {
                auto residual = output;
                auto branch = output;

                for (const auto& layer : layers) {
                    branch = layer.forward(std::move(branch));
                    branch = ::Omni::Activation::Details::apply(layer.activation, std::move(branch));
                }

                auto skip_connection = residual;
                if (projection_layer_.has_value()) {
                    skip_connection = projection_layer_->forward(std::move(skip_connection));
                    skip_connection = ::Omni::Activation::Details::apply(
                        projection_layer_->activation, std::move(skip_connection));
                }

                if (branch.sizes() != skip_connection.sizes()) {
                    auto format_shape = [](const torch::Tensor& tensor) {
                        std::ostringstream stream;
                        stream << '(';
                        bool first = true;
                        for (const auto dimension : tensor.sizes()) {
                            if (!first) {
                                stream << ", ";
                            }
                            first = false;
                            stream << dimension;
                        }
                        stream << ')';
                        return stream.str();
                    };

                    std::ostringstream message;
                    message << "Residual block skip connection shape mismatch: branch output "
                            << format_shape(branch)
                            << " vs. skip connection "
                            << format_shape(skip_connection)
                            << ". Consider providing a projection layer or adjusting the block configuration.";
                    throw std::runtime_error(message.str());
                }

                branch = std::move(branch) + std::move(skip_connection);
                branch = ::Omni::Activation::Details::apply(final_activation_, std::move(branch));
                if (dropout_) {
                    branch = dropout_->forward(branch);
                }

                output = std::move(branch);
            }

            return output;
        }

    private:
        template <typename ConvolutionImpl>
       void apply_tensor_memory_format_to_convolution(ConvolutionImpl* convolution)
        {
            if (!convolution) {
                return;
            }

            auto apply_to_parameter = [&](torch::Tensor& parameter) {
                if (!parameter.defined()) {
                    return;
                }

                parameter = parameter.to(
                 parameter.options(),
                 /*non_blocking=*/false,
                 /*copy=*/false,
                 preferred_tensor_memory_format_);
            };

            apply_to_parameter(convolution->weight);
            if (convolution->bias.defined()) {
                apply_to_parameter(convolution->bias);
            }
        }

        void apply_tensor_memory_format_to_layer(::Omni::Layer::Details::RegisteredLayer& layer)
        {
            if (!layer.module) {
                return;
            }

            if (auto* conv1d = dynamic_cast<torch::nn::Conv1dImpl*>(layer.module.get())) {
                apply_tensor_memory_format_to_convolution(conv1d);
            } else if (auto* conv2d = dynamic_cast<torch::nn::Conv2dImpl*>(layer.module.get())) {
                apply_tensor_memory_format_to_convolution(conv2d);
            }
        }

        void apply_tensor_memory_format_to_convolutions()
        {
            for (auto& layers : block_layers_) {
                for (auto& layer : layers) {
                    apply_tensor_memory_format_to_layer(layer);
                }
            }

            if (projection_layer_) {
                apply_tensor_memory_format_to_layer(*projection_layer_);
            }
        }
        std::vector<std::vector<::Omni::Layer::Details::RegisteredLayer>> block_layers_{};
        std::optional<::Omni::Layer::Details::RegisteredLayer> projection_layer_{};
        ::Omni::Activation::Type final_activation_{::Omni::Activation::Type::Identity};
        torch::nn::Dropout dropout_{nullptr};
        torch::MemoryFormat preferred_tensor_memory_format_{torch::MemoryFormat::Contiguous};
    };

    TORCH_MODULE(ResidualBlock);
}
#endif // OMNI_BLOCK_DETAILS_RESIDUAL_HPP
