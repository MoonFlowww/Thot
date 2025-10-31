#ifndef THOT_RECURRENT_HPP
#define THOT_RECURRENT_HPP

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <tuple>

#include <torch/torch.h>

#include "../../activation/activation.hpp"
#include "../../common/local.hpp"
#include "../../initialization/initialization.hpp"
#include "../registry.hpp"


namespace Thot::Layer::Details {

    struct RNNOptions {
        std::int64_t input_size{};
        std::int64_t hidden_size{};
        std::int64_t num_layers{1};
        double dropout{0.0};
        bool batch_first{false};
        bool bidirectional{false};
        std::string nonlinearity{"tanh"};
    };

    struct LSTMOptions {
        std::int64_t input_size{};
        std::int64_t hidden_size{};
        std::int64_t num_layers{1};
        double dropout{0.0};
        bool batch_first{false};
        bool bidirectional{false};
    };

    struct GRUOptions {
        std::int64_t input_size{};
        std::int64_t hidden_size{};
        std::int64_t num_layers{1};
        double dropout{0.0};
        bool batch_first{false};
        bool bidirectional{false};
    };

    struct RNNDescriptor {
        RNNOptions options{};
        ::Thot::Activation::Descriptor activation{::Thot::Activation::Identity};
        ::Thot::Initialization::Descriptor initialization{::Thot::Initialization::Default};
        ::Thot::LocalConfig local{};
    };

    struct LSTMDescriptor {
        LSTMOptions options{};
        ::Thot::Activation::Descriptor activation{::Thot::Activation::Identity};
        ::Thot::Initialization::Descriptor initialization{::Thot::Initialization::Default};
        ::Thot::LocalConfig local{};
    };

    struct GRUDescriptor {
        GRUOptions options{};
        ::Thot::Activation::Descriptor activation{::Thot::Activation::Identity};
        ::Thot::Initialization::Descriptor initialization{::Thot::Initialization::Default};
        ::Thot::LocalConfig local{};
    };

    namespace Detail {

        template <class ModuleHolder, class Descriptor>
        void apply_recurrent_initialization(const ModuleHolder& module, const Descriptor& descriptor)
        {
            const auto type = descriptor.initialization.type;
            if (type == ::Thot::Initialization::Type::Default) {
                return;
            }

            auto parameters = module->named_parameters(/*recurse=*/true);
            for (auto& named_parameter : parameters) {
                auto& parameter = named_parameter.value();
                if (!parameter.defined()) {
                    continue;
                }

                const auto& name = named_parameter.key();
                const bool is_bias = name.find("bias") != std::string::npos;
                const bool is_weight = name.find("weight") != std::string::npos;

                if (is_weight) {
                    switch (type) {
                        case ::Thot::Initialization::Type::XavierNormal:
                            torch::nn::init::xavier_normal_(parameter);
                            break;
                        case ::Thot::Initialization::Type::XavierUniform:
                            torch::nn::init::xavier_uniform_(parameter);
                            break;
                        case ::Thot::Initialization::Type::HeNormal:
                            torch::nn::init::kaiming_normal_(parameter, /*a=*/0.0, torch::kFanIn, torch::kReLU);
                            break;
                        case ::Thot::Initialization::Type::HeUniform:
                            torch::nn::init::kaiming_uniform_(parameter, /*a=*/0.0, torch::kFanIn, torch::kReLU);
                            break;
                        case ::Thot::Initialization::Type::Dirac:
                            if (parameter.dim() >= 3) {
                                torch::nn::init::dirac_(parameter);
                            }
                            break;
                        case ::Thot::Initialization::Type::Lyapunov:
                            torch::nn::init::orthogonal_(parameter);
                            break;
                        case ::Thot::Initialization::Type::ZeroBias:
                        case ::Thot::Initialization::Type::Default:
                        default:
                            break;
                    }
                }

                if (is_bias && (type == ::Thot::Initialization::Type::XavierNormal ||
                                 type == ::Thot::Initialization::Type::XavierUniform ||
                                 type == ::Thot::Initialization::Type::HeNormal ||
                                 type == ::Thot::Initialization::Type::HeUniform ||
                                 type == ::Thot::Initialization::Type::Lyapunov ||
                                 type == ::Thot::Initialization::Type::ZeroBias)) {
                    torch::nn::init::zeros_(parameter);
                }
            }
        }

        template <class Options>
        void validate_recurrent_options(const Options& options, const char* name)
        {
            if (options.input_size <= 0 || options.hidden_size <= 0) {
                throw std::invalid_argument(std::string(name) + " layers require positive input and hidden sizes.");
            }
            if (options.num_layers <= 0) {
                throw std::invalid_argument(std::string(name) + " layers require at least one layer.");
            }
            if (options.dropout < 0.0 || options.dropout >= 1.0) {
                throw std::invalid_argument(std::string(name) + " dropout must be in [0, 1).");
            }
        }

        inline torch::nn::RNNOptions::nonlinearity_t normalize_nonlinearity(std::string nonlinearity) {
            std::transform(nonlinearity.begin(), nonlinearity.end(), nonlinearity.begin(),
                           [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
            if (nonlinearity == "relu") {
                return torch::kReLU;
            }
            if (nonlinearity == "tanh") {
                return torch::kTanh;
            }
            throw std::invalid_argument("RNN nonlinearity must be either 'tanh' or 'relu'.");
        }

        template <class Output>
        auto take_recurrent_output(Output&& result)
        {
            if constexpr (requires { std::forward<Output>(result).output; }) {
                return std::forward<Output>(result).output;
            } else if constexpr (requires { std::get<0>(std::forward<Output>(result)); }) {
                return std::get<0>(std::forward<Output>(result));
            } else {
                static_assert(sizeof(Output) == 0, "Unsupported recurrent module output type.");
            }
        }


        template <class Owner, class ModuleType, class Descriptor, class Options>
        RegisteredLayer build_recurrent_layer(Owner& owner,
                                              std::string name_prefix,
                                              const Descriptor& descriptor,
                                              std::size_t index,
                                              Options options)
        {
            if (descriptor.options.num_layers > 1 && descriptor.options.dropout > 0.0) {
                options.dropout(descriptor.options.dropout);
            }

            auto module = owner.register_module(std::move(name_prefix) + std::to_string(index), ModuleType(options));
            Detail::apply_recurrent_initialization(module, descriptor);

            RegisteredLayer registered_layer{};
            registered_layer.activation = descriptor.activation.type;
            registered_layer.module = to_shared_module_ptr(module);
            registered_layer.local = descriptor.local;
            struct RecurrentForwardFunctor {
                decltype(module.get()) module_ptr;

                torch::Tensor operator()(torch::Tensor input) const
                {
                    auto output = module_ptr->forward(std::move(input));
                    return Detail::take_recurrent_output(std::move(output));
                }
            };
            registered_layer.bind_inline_forward(RecurrentForwardFunctor{module.get()});
            return registered_layer;
        }

    }

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const RNNDescriptor& descriptor, std::size_t index)
    {
        Detail::validate_recurrent_options(descriptor.options, "RNN");
        auto nonlinearity = Detail::normalize_nonlinearity(descriptor.options.nonlinearity);

        auto options = torch::nn::RNNOptions(descriptor.options.input_size, descriptor.options.hidden_size)
                           .num_layers(descriptor.options.num_layers)
                           .batch_first(descriptor.options.batch_first)
                           .bidirectional(descriptor.options.bidirectional)
                           .nonlinearity(nonlinearity);

        return Detail::build_recurrent_layer<Owner, torch::nn::RNN>(owner, "rnn_", descriptor, index, options);
    }

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const LSTMDescriptor& descriptor, std::size_t index)
    {
        Detail::validate_recurrent_options(descriptor.options, "LSTM");

        auto options = torch::nn::LSTMOptions(descriptor.options.input_size, descriptor.options.hidden_size)
                           .num_layers(descriptor.options.num_layers)
                           .batch_first(descriptor.options.batch_first)
                           .bidirectional(descriptor.options.bidirectional);

        return Detail::build_recurrent_layer<Owner, torch::nn::LSTM>(owner, "lstm_", descriptor, index, options);
    }

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const GRUDescriptor& descriptor, std::size_t index)
    {
        Detail::validate_recurrent_options(descriptor.options, "GRU");

        auto options = torch::nn::GRUOptions(descriptor.options.input_size, descriptor.options.hidden_size)
                           .num_layers(descriptor.options.num_layers)
                           .batch_first(descriptor.options.batch_first)
                           .bidirectional(descriptor.options.bidirectional);

        return Detail::build_recurrent_layer<Owner, torch::nn::GRU>(owner, "gru_", descriptor, index, options);
    }

}

#endif //THOT_RECURRENT_HPP