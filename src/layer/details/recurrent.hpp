#ifndef THOT_RECURRENT_HPP
#define THOT_RECURRENT_HPP

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <tuple>
#include <vector>

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

    // ======================= xLSTM (Extended LSTM) API =========================
    struct xLSTMOptions {
        std::int64_t input_size{};
        std::int64_t hidden_size{};
        std::int64_t num_layers{1};
        double dropout{0.0};
        bool batch_first{false};
        bool bidirectional{false};

        // x-extensions
        std::int64_t proj_size{0};          // 0 => no projection; otherwise output dim per direction
        bool layer_norm_gates{false};       // LN over concatenated gates (i,f,g,o)
        bool layer_norm_state{false};       // LN over cell state c_t
        double ln_eps{1e-5};
        bool use_mi{true};                  // multiplicative integration
        double forget_bias{1.0};            // bias term added to forget gate pre-activation (applied once at init)
    };

    struct xLSTMDescriptor {
        xLSTMOptions options{};
        ::Thot::Activation::Descriptor activation{::Thot::Activation::Identity};
        ::Thot::Initialization::Descriptor initialization{::Thot::Initialization::Default};
        ::Thot::LocalConfig local{};
    };
    // ===================== end xLSTM API =====================

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

        // ======================== xLSTM implementation ========================

        // One cell with optional MI, gate/state LN, and projection.
        struct xLSTMCellImpl : torch::nn::Module {
            const std::int64_t input_size;
            const std::int64_t hidden_size;
            const std::int64_t proj_size;
            const bool ln_gates;
            const bool ln_state;
            const double ln_eps;
            const bool use_mi;
            const double forget_bias;

            torch::nn::Linear w_ih{nullptr}; // [I -> 4H], bias=false
            torch::nn::Linear w_hh{nullptr}; // [h_in -> 4H], bias=false; h_in = (proj ? proj : H)
            torch::Tensor alpha, beta, gamma; // MI scalars per dim (4H)
            torch::Tensor bias;               // (4H)

            torch::nn::LayerNorm ln_g{nullptr};   // over 4H
            torch::nn::LayerNorm ln_c{nullptr};   // over H
            torch::nn::Linear proj{nullptr};      // optional H -> P

            explicit xLSTMCellImpl(const xLSTMOptions& opt)
                : input_size(opt.input_size)
                , hidden_size(opt.hidden_size)
                , proj_size(opt.proj_size)
                , ln_gates(opt.layer_norm_gates)
                , ln_state(opt.layer_norm_state)
                , ln_eps(opt.ln_eps)
                , use_mi(opt.use_mi)
                , forget_bias(opt.forget_bias)
            {
                const auto h_in = (proj_size > 0) ? proj_size : hidden_size;

                w_ih = register_module("w_ih", torch::nn::Linear(
                    torch::nn::LinearOptions(input_size, 4 * hidden_size).bias(false)));
                w_hh = register_module("w_hh", torch::nn::Linear(
                    torch::nn::LinearOptions(h_in, 4 * hidden_size).bias(false)));

                alpha = register_parameter("alpha", torch::ones({4 * hidden_size}));
                beta  = register_parameter("beta",  torch::ones({4 * hidden_size}));
                gamma = register_parameter("gamma", torch::ones({4 * hidden_size}));
                bias  = register_parameter("bias",  torch::zeros({4 * hidden_size}));

                // Apply forget bias once at init on the forget gate slice [H..2H)
                if (forget_bias != 0.0) {
                    bias.slice(/*dim=*/0, hidden_size, 2 * hidden_size).add_(forget_bias);
                }

                if (ln_gates) {
                    ln_g = register_module("ln_g",
                        torch::nn::LayerNorm(torch::nn::LayerNormOptions({4 * hidden_size}).eps(ln_eps)));
                }
                if (ln_state) {
                    ln_c = register_module("ln_c",
                        torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size}).eps(ln_eps)));
                }
                if (proj_size > 0) {
                    proj = register_module("proj",
                        torch::nn::Linear(torch::nn::LinearOptions(hidden_size, proj_size).bias(true)));
                }
            }

            // x: [B, I], h_prev: [B, h_in], c_prev: [B, H]
            std::pair<torch::Tensor, torch::Tensor>
            step(const torch::Tensor& x, const torch::Tensor& h_prev, const torch::Tensor& c_prev)
            {
                auto U = w_ih->forward(x);        // [B, 4H]
                auto V = w_hh->forward(h_prev);   // [B, 4H]

                torch::Tensor pre;
                if (use_mi) {
                    // Fuse elementwise ops to reduce temporaries
                    auto a = alpha.unsqueeze(0);
                    auto b = beta.unsqueeze(0);
                    auto g = gamma.unsqueeze(0);
                    pre = U.mul(V);           // U*V
                    pre.mul_(a);              // a*(U*V)
                    pre.add_(U.mul(b));       // + b*U
                    pre.add_(V.mul(g));       // + g*V
                    pre.add_(bias.unsqueeze(0));
                } else {
                    pre = U + V + bias.unsqueeze(0);
                }

                if (ln_gates) pre = ln_g->forward(pre);

                auto chunks = pre.chunk(4, /*dim=*/1);
                auto i = torch::sigmoid(chunks[0]);
                auto f = torch::sigmoid(chunks[1]); // forget bias already baked into bias param
                auto gg = torch::tanh(chunks[2]);
                auto o = torch::sigmoid(chunks[3]);

                auto c_t = f * c_prev + i * gg;
                if (ln_state) c_t = ln_c->forward(c_t);

                auto m_t = torch::tanh(c_t);
                auto h_raw = o * m_t; // [B, H]
                torch::Tensor h_t_out = (proj_size > 0) ? proj->forward(h_raw) : h_raw;
                return {h_t_out, c_t};
            }
        };
        TORCH_MODULE(xLSTMCell);

        // One direction, one layer (time-major input [S,B,I]).
        struct xLSTMLayerImpl : torch::nn::Module {
            xLSTMCell cell;

            explicit xLSTMLayerImpl(const xLSTMOptions& opt)
            : cell(register_module("cell", xLSTMCell(opt))) {}

            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
            forward_tb(const torch::Tensor& X, torch::Tensor h0, torch::Tensor c0)
            {
                const auto S = X.size(0);
                const auto B = X.size(1);
                const auto dev = X.device();
                const auto dt = X.dtype();
                const auto h_out = (cell->proj_size > 0) ? cell->proj_size : cell->hidden_size;

                auto Y = torch::empty({S, B, h_out}, torch::TensorOptions().dtype(dt).device(dev));

                auto h_t = std::move(h0);
                auto c_t = std::move(c0);

                for (int64_t t = 0; t < S; ++t) {
                    auto hc = cell->step(X[t], h_t, c_t);
                    h_t = hc.first;
                    c_t = hc.second;
                    Y.select(0, t).copy_(h_t);
                }
                return {Y, h_t, c_t};
            }

            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
            forward_bt(const torch::Tensor& X, torch::Tensor h0, torch::Tensor c0)
            {
                const auto S = X.size(0);
                const auto B = X.size(1);
                const auto dev = X.device();
                const auto dt = X.dtype();
                const auto h_out = (cell->proj_size > 0) ? cell->proj_size : cell->hidden_size;

                auto Y = torch::empty({S, B, h_out}, torch::TensorOptions().dtype(dt).device(dev));

                auto h_t = std::move(h0);
                auto c_t = std::move(c0);

                for (int64_t idx = S - 1; idx >= 0; --idx) {
                    auto hc = cell->step(X[idx], h_t, c_t);
                    h_t = hc.first;
                    c_t = hc.second;
                    Y.select(0, idx).copy_(h_t);
                }
                return {Y, h_t, c_t};
            }
        };
        TORCH_MODULE(xLSTMLayer);

        // Stacked + (optional) bidirectional xLSTM with cuDNN fast-path.
        struct xLSTMImpl : torch::nn::Module {
            const xLSTMOptions opt;
            const int64_t num_dir;
            const int64_t h_out; // per-direction output size = (proj?proj:hidden)
            const int64_t h_in;  // input size to recurrent W_hh = h_out

            std::vector<xLSTMLayer> layers_fwd;
            std::vector<xLSTMLayer> layers_bwd; // only if bidirectional

            // cuDNN fast-path when no x-features are used
            torch::nn::LSTM cudnn_lstm{nullptr};
            bool use_cudnn_fastpath{false};

            explicit xLSTMImpl(xLSTMOptions options)
            : opt(std::move(options))
            , num_dir(opt.bidirectional ? 2 : 1)
            , h_out((opt.proj_size > 0) ? opt.proj_size : opt.hidden_size)
            , h_in(h_out)
            {
                use_cudnn_fastpath = (!opt.use_mi &&
                                      !opt.layer_norm_gates &&
                                      !opt.layer_norm_state &&
                                      opt.proj_size == 0 &&
                                      std::abs(opt.forget_bias - 1.0) < 1e-12);

                if (use_cudnn_fastpath) {
                    auto o = torch::nn::LSTMOptions(opt.input_size, opt.hidden_size)
                                .num_layers(opt.num_layers)
                                .batch_first(opt.batch_first)
                                .bidirectional(opt.bidirectional)
                                .dropout(opt.dropout);
                    cudnn_lstm = register_module("cudnn_lstm", torch::nn::LSTM(o));
                    return;
                }

                int64_t in_dim = opt.input_size;
                for (int64_t l = 0; l < opt.num_layers; ++l) {
                    xLSTMOptions local = opt;
                    local.input_size = (l == 0) ? in_dim : (h_out * num_dir);
                    layers_fwd.emplace_back(register_module("layer_fwd_" + std::to_string(l), xLSTMLayer(local)));
                    if (opt.bidirectional) {
                        layers_bwd.emplace_back(register_module("layer_bwd_" + std::to_string(l), xLSTMLayer(local)));
                    }
                }
            }

            // Returns (output, h_n, c_n) like torch::nn::LSTM
            // input: [B,S,I] if batch_first else [S,B,I]
            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
            forward(const torch::Tensor& input)
            {
                if (use_cudnn_fastpath) {
                    auto [Y, hx] = cudnn_lstm->forward(input);
                    auto [h_n, c_n] = hx;
                    return std::make_tuple(std::move(Y), std::move(h_n), std::move(c_n));
                }


                torch::Tensor X = input;
                const bool bf = opt.batch_first;
                if (bf) X = X.transpose(0, 1); // [S,B,I]

                const auto B = X.size(1);
                const auto dev = X.device();
                const auto dt = X.dtype();

                torch::Tensor Y = X;
                std::vector<torch::Tensor> h_final; // per layer*dir: [B,h_out]
                std::vector<torch::Tensor> c_final; // per layer*dir: [B,H]

                for (int64_t l = 0; l < opt.num_layers; ++l) {
                    // initial states per direction (reused, no clones)
                    auto zeros_h = torch::zeros({B, h_in}, torch::TensorOptions().dtype(dt).device(dev));
                    auto zeros_c = torch::zeros({B, opt.hidden_size}, torch::TensorOptions().dtype(dt).device(dev));

                    torch::Tensor Y_fwd, hT_fwd, cT_fwd;
                    std::tie(Y_fwd, hT_fwd, cT_fwd) = layers_fwd[l]->forward_tb(Y, zeros_h, zeros_c);

                    torch::Tensor Y_cat;
                    if (opt.bidirectional) {
                        torch::Tensor Y_bwd, hT_bwd, cT_bwd;
                        std::tie(Y_bwd, hT_bwd, cT_bwd) = layers_bwd[l]->forward_bt(Y, zeros_h, zeros_c);
                        Y_cat = torch::cat({Y_fwd, Y_bwd}, /*dim=*/2); // concat on feature
                        h_final.emplace_back(hT_fwd);
                        h_final.emplace_back(hT_bwd);
                        c_final.emplace_back(cT_fwd);
                        c_final.emplace_back(cT_bwd);
                    } else {
                        Y_cat = Y_fwd;
                        h_final.emplace_back(hT_fwd);
                        c_final.emplace_back(cT_fwd);
                    }

                    // Dropout between layers (like PyTorch) except after last
                    if (opt.dropout > 0.0 && l < opt.num_layers - 1) {
                        Y = torch::dropout(Y_cat, /*p=*/opt.dropout, /*train=*/is_training());
                    } else {
                        Y = std::move(Y_cat);
                    }
                }

                const int64_t LDN = opt.num_layers * num_dir;
                const auto B2 = Y.size(1);
                auto h_n = torch::stack(h_final, /*dim=*/0).view({LDN, B2, h_out});         // [LDN, B, h_out]
                auto c_n = torch::stack(c_final, /*dim=*/0).view({LDN, B2, opt.hidden_size}); // [LDN, B, H]

                if (bf) {
                    Y = Y.transpose(0, 1); // [B,S,num_dir*h_out]
                }
                return {Y, h_n, c_n};
            }
        };
        TORCH_MODULE(xLSTM);

        // ====================== end xLSTM implementation ======================

    } // namespace Detail

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
                           .bidirectional(descriptor.options.bidirectional)
                           .dropout(descriptor.options.dropout);

        return Detail::build_recurrent_layer<Owner, torch::nn::LSTM>(owner, "lstm_", descriptor, index, options);
    }

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const GRUDescriptor& descriptor, std::size_t index)
    {
        Detail::validate_recurrent_options(descriptor.options, "GRU");

        auto options = torch::nn::GRUOptions(descriptor.options.input_size, descriptor.options.hidden_size)
                           .num_layers(descriptor.options.num_layers)
                           .batch_first(descriptor.options.batch_first)
                           .bidirectional(descriptor.options.bidirectional)
                           .dropout(descriptor.options.dropout);

        return Detail::build_recurrent_layer<Owner, torch::nn::GRU>(owner, "gru_", descriptor, index, options);
    }

    // ----------------------- Builder for xLSTM ------------------------
    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const xLSTMDescriptor& descriptor, std::size_t index)
    {
        Detail::validate_recurrent_options(descriptor.options, "xLSTM");

        auto module = owner.register_module("xlstm_" + std::to_string(index),
                                            Detail::xLSTM(descriptor.options));

        Detail::apply_recurrent_initialization(module, descriptor);

        RegisteredLayer registered_layer{};
        registered_layer.activation = descriptor.activation.type;
        registered_layer.module = to_shared_module_ptr(module);
        registered_layer.local = descriptor.local;

        struct ForwardFunctor {
            decltype(module.get()) module_ptr;
            torch::Tensor operator()(torch::Tensor input) const {
                auto out = module_ptr->forward(std::move(input));
                return Detail::take_recurrent_output(std::move(out));
            }
        };
        registered_layer.bind_inline_forward(ForwardFunctor{module.get()});
        return registered_layer;
    }

}
#endif //THOT_RECURRENT_HPP
