#ifndef THOT_STATESPACE_HPP
#define THOT_STATESPACE_HPP

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "../../activation/activation.hpp"
#include "../../common/local.hpp"
#include "../../initialization/apply.hpp"
#include "../../initialization/initialization.hpp"
#include "../registry.hpp"


namespace Thot::Layer::Details {
    struct StateSpaceOptions {
        std::int64_t input_size{};
        std::int64_t hidden_size{};
        std::int64_t output_size{};
        std::int64_t num_layers{1};
        double dropout{0.0};
        bool batch_first{false};
        bool bidirectional{false};
    };

    struct StateSpaceDescriptor {
        StateSpaceOptions options{};
        ::Thot::Activation::Descriptor activation{::Thot::Activation::Identity};
        ::Thot::Initialization::Descriptor initialization{::Thot::Initialization::Default};
        ::Thot::LocalConfig local{};
    };

    struct StateSpaceOutput {
        torch::Tensor output{};
        torch::Tensor state{};
    };

    class StateSpaceModuleImpl : public torch::nn::Module {
    public:
        using Options = StateSpaceOptions;

        explicit StateSpaceModuleImpl(Options options)
            : options_(std::move(options))
        {
            const auto directions = options_.bidirectional ? 2 : 1;
            if (options_.num_layers <= 0) {
                throw std::invalid_argument("StateSpaceModule requires at least one layer.");
            }
            if (options_.hidden_size <= 0) {
                throw std::invalid_argument("StateSpaceModule requires a positive hidden size.");
            }
            if (options_.input_size <= 0) {
                throw std::invalid_argument("StateSpaceModule requires a positive input size.");
            }
            if (options_.output_size <= 0) {
                options_.output_size = options_.hidden_size;
            }
            if (options_.dropout < 0.0 || options_.dropout >= 1.0) {
                throw std::invalid_argument("StateSpaceModule dropout must be in [0, 1).");
            }

            const auto total_blocks = options_.num_layers * directions;
            input_linears_.reserve(static_cast<std::size_t>(total_blocks));
            state_linears_.reserve(static_cast<std::size_t>(total_blocks));
            output_linears_.reserve(static_cast<std::size_t>(total_blocks));

            for (std::int64_t layer = 0; layer < options_.num_layers; ++layer) {
                for (std::int64_t direction = 0; direction < directions; ++direction) {
                    const std::int64_t index = layer * directions + direction;
                    const auto layer_input_size = layer == 0
                        ? options_.input_size
                        : options_.output_size * directions;

                    auto input_linear = register_module(
                        "input_linear_" + std::to_string(index),
                        torch::nn::Linear(torch::nn::LinearOptions(layer_input_size, options_.hidden_size)));
                    auto state_linear = register_module(
                        "state_linear_" + std::to_string(index),
                        torch::nn::Linear(torch::nn::LinearOptions(options_.hidden_size, options_.hidden_size).bias(false)));
                    auto output_linear = register_module(
                        "output_linear_" + std::to_string(index),
                        torch::nn::Linear(torch::nn::LinearOptions(options_.hidden_size, options_.output_size)));

                    input_linears_.push_back(input_linear);
                    state_linears_.push_back(state_linear);
                    output_linears_.push_back(output_linear);
                }
            }

            if (options_.dropout > 0.0 && options_.num_layers > 1) {
                dropout_ = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(options_.dropout)));
            }
        }

        [[nodiscard]] StateSpaceOutput forward(torch::Tensor input, torch::Tensor state = {})
        {
            auto tensor = std::move(input);
            if (!tensor.defined()) {
                throw std::invalid_argument("StateSpaceModule requires a defined input tensor.");
            }

            if (tensor.dim() != 3) {
                throw std::invalid_argument("StateSpaceModule expects inputs shaped as (seq, batch, feature) or (batch, seq, feature).");
            }

            if (options_.batch_first) {
                tensor = tensor.transpose(0, 1);
            }

            const auto seq_len = tensor.size(0);
            const auto batch = tensor.size(1);
            const auto directions = options_.bidirectional ? 2 : 1;
            const auto layers = options_.num_layers;

            auto working_state = state;
            if (!working_state.defined() || working_state.numel() == 0) {
                working_state = torch::zeros({layers * directions, batch, options_.hidden_size}, tensor.options());
            }

            auto new_state = torch::zeros_like(working_state);
            auto layer_input = tensor;

            for (std::int64_t layer = 0; layer < layers; ++layer) {
                std::vector<torch::Tensor> direction_outputs;
                direction_outputs.reserve(static_cast<std::size_t>(directions));

                for (std::int64_t direction = 0; direction < directions; ++direction) {
                    const std::int64_t block_index = layer * directions + direction;
                    auto hx = working_state[block_index];
                    std::vector<torch::Tensor> emissions(static_cast<std::size_t>(seq_len));

                    auto& input_linear = input_linears_[static_cast<std::size_t>(block_index)];
                    auto& state_linear = state_linears_[static_cast<std::size_t>(block_index)];
                    auto& output_linear = output_linears_[static_cast<std::size_t>(block_index)];

                    for (std::int64_t step = 0; step < seq_len; ++step) {
                        const std::int64_t timestep = direction == 0 ? step : (seq_len - 1 - step);
                        auto current_input = layer_input[timestep];
                        auto projected = input_linear->forward(current_input) + state_linear->forward(hx);
                        auto next_state = torch::tanh(projected);
                        auto emission = output_linear->forward(next_state);

                        emissions[static_cast<std::size_t>(timestep)] = std::move(emission);
                        hx = std::move(next_state);
                    }

                    direction_outputs.push_back(torch::stack(emissions));
                    new_state[block_index] = hx;
                }

                torch::Tensor combined = direction_outputs.size() == 1
                    ? direction_outputs.front()
                    : torch::cat(direction_outputs, /*dim=*/-1);

                if (dropout_ && layer + 1 < layers) {
                    combined = dropout_->forward(combined);
                }

                layer_input = combined;
            }

            if (options_.batch_first) {
                layer_input = layer_input.transpose(0, 1);
            }

            return {std::move(layer_input), std::move(new_state)};
        }

        void apply_initialization(const ::Thot::Initialization::Descriptor& descriptor)
        {
            struct ProxyDescriptor {
                ::Thot::Initialization::Descriptor initialization;
            };
            ProxyDescriptor proxy{descriptor};
            for (auto& module : input_linears_) {
                ::Thot::Initialization::Details::apply_module_initialization(module, proxy);
            }
            for (auto& module : state_linears_) {
                ::Thot::Initialization::Details::apply_module_initialization(module, proxy);
            }
            for (auto& module : output_linears_) {
                ::Thot::Initialization::Details::apply_module_initialization(module, proxy);
            }
        }

        [[nodiscard]] const Options& options() const noexcept { return options_; }

    private:
        Options options_{};
        std::vector<torch::nn::Linear> input_linears_{};
        std::vector<torch::nn::Linear> state_linears_{};
        std::vector<torch::nn::Linear> output_linears_{};
        torch::nn::Dropout dropout_{nullptr};
    };

    TORCH_MODULE(StateSpaceModule);

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const StateSpaceDescriptor& descriptor, std::size_t index)
    {
        auto options = descriptor.options;
        if (options.input_size <= 0 || options.hidden_size <= 0) {
            throw std::invalid_argument("State-space layers require positive input and hidden sizes.");
        }
        if (options.num_layers <= 0) {
            throw std::invalid_argument("State-space layers require at least one layer.");
        }
        if (options.output_size <= 0) {
            options.output_size = options.hidden_size;
        }
        if (options.dropout < 0.0 || options.dropout >= 1.0) {
            throw std::invalid_argument("State-space dropout must be in [0, 1).");
        }

        auto module = owner.register_module("statespace_" + std::to_string(index), StateSpaceModule(options));
        module->apply_initialization(descriptor.initialization);

        RegisteredLayer registered_layer{};
        registered_layer.activation = descriptor.activation.type;
        registered_layer.module = to_shared_module_ptr(module);
        registered_layer.local = descriptor.local;
        registered_layer.forward = [module](torch::Tensor input) {
            auto output = module->forward(std::move(input));
            return std::move(output.output);
        };
        return registered_layer;
    }
}

#endif // THOT_STATESPACE_HPP