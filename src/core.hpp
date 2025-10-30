#ifndef THOT_CORE_HPP
#define THOT_CORE_HPP
/*
 * Core orchestrator of the framework.
 * ---------------------------------------------------------------------------
 * Planned responsibilities:
 *  - Define the master constexpr configuration (model topology, data pipeline,
 *    optimization, metrics, regularization) that remains immutable at runtime.
 *  - Collect requests from main.cpp and route them to the appropriate
 *    factories located under the module directories (activation, block, layer,
 *    optimizer, etc.).
 *  - Materialise the selected components as inline function objects or raw
 *    pointers that can be handed over to network.hpp without leaking
 *    higher-level orchestration details.
 *  - Push only the latency-inflating feature toggles (regularisation, data
 *    augmentation, k-folding, etc.) behind `if constexpr` / template
 *    specialisation so the generated runtime code path is branchless once the
 *    compile-time configuration is instantiated; constant-cost utilities can
 *    stay in regular functions for readability.
 *  - Expose helper APIs to retrieve training, evaluation, calibration and
 *    monitoring routines pre-bound to the compile-time configuration.
 */

#include <algorithm>
#include <functional>
#include <chrono>
#include <cstddef>
#include <initializer_list>
#include <cmath>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>
#include <deque>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <cctype>
#include <iterator>


#include <torch/torch.h>



#include "common/save_load.hpp"
#include "utils/terminal.hpp"
#include "activation/activation.hpp"
#include "activation/apply.hpp"
#include "initialization/initialization.hpp"
#include "attention/details/head.hpp"
#include "block/block.hpp"
#include "initialization/apply.hpp"
#include "layer/layer.hpp"
#include "block/details/blocks/residual.hpp"
#include "evaluation/evaluation.hpp"
#include "loss/loss.hpp"
#include "loss/details/mse.hpp"
#include "optimizer/optimizer.hpp"
#include "lrscheduler/lrscheduler.hpp"
#include "block/block.hpp"
#include "layer/details/positional_encoding.hpp"

#include "../src/regularization/regularization.hpp"
#include "../src/regularization/apply.hpp"
#include "calibration/calibration.hpp"

namespace Thot {
    template <class... Ts>
    struct Overloaded : Ts... {
        using Ts::operator()...;
    };

    template <class... Ts>
    Overloaded(Ts...) -> Overloaded<Ts...>;

    enum class MergePolicyKind {
        Strict,
        Broadcast,
        Concat
    };

    struct Port {
        enum class Kind {
            Input,
            Output,
            Module,
            Join
        };

        Kind kind{Kind::Module};
        std::string identifier{};
        std::string attribute{};
        std::string representation{};
        MergePolicyKind merge_policy{MergePolicyKind::Strict};
        std::optional<std::size_t> node_index{};
        std::optional<std::size_t> join_index{};
        std::vector<std::string> join_members{};
        std::optional<int64_t> join_dimension{};

        Port() = default;

        Port(Kind kind, std::string identifier, std::string attribute, MergePolicyKind merge)
            : kind(kind),
              identifier(std::move(identifier)),
              attribute(std::move(attribute)),
              representation(build_representation()),
              merge_policy(merge)
        {
        }

        [[nodiscard]] bool is_input() const noexcept { return kind == Kind::Input; }
        [[nodiscard]] bool is_output() const noexcept { return kind == Kind::Output; }
        [[nodiscard]] bool is_module() const noexcept { return kind == Kind::Module; }
        [[nodiscard]] bool is_join() const noexcept { return kind == Kind::Join; }
        [[nodiscard]] const std::string& describe() const noexcept { return representation; }
        [[nodiscard]] std::string storage_key() const
        {
            if (!join_members.empty()) {
                std::string key = identifier;
                if (!attribute.empty()) {
                    key.append(":");
                    key.append(attribute);
                }
                return key;
            }
            if (attribute.empty()) {
                return identifier;
            }
            return identifier + ":" + attribute;
        }

        void assign_node(std::size_t value) { node_index = value; }
        void assign_join(std::size_t value) { join_index = value; }

        static Port parse(std::string_view specification)
        {
            const auto trimmed = trim(specification);
            if (trimmed.empty()) {
                throw std::invalid_argument("Port::parse requires a non-empty specification.");
            }

            auto [token, attribute] = split_token(trimmed);
            token = trim(token);
            attribute = trim(attribute);

            Port port{};
            port.attribute.assign(attribute.begin(), attribute.end());
            port.representation.assign(trimmed.begin(), trimmed.end());

            if (token.empty()) {
                throw std::invalid_argument("Port::parse encountered an empty identifier in '" + port.representation + "'.");
            }

            if (token.front() == '@') {
                auto sentinel = token.substr(1);
                if (sentinel == "input") {
                    port.kind = Kind::Input;
                } else if (sentinel == "output") {
                    port.kind = Kind::Output;
                } else {
                    throw std::invalid_argument("Unsupported sentinel port '@" + std::string(sentinel) + "'.");
                }
                port.identifier.assign(sentinel.begin(), sentinel.end());
            } else {
                port.kind = Kind::Module;
                port.identifier.assign(token.begin(), token.end());
            }

            if (port.identifier.empty()) {
                throw std::invalid_argument("Port specification '" + port.representation + "' is missing an identifier.");
            }

            port.merge_policy = MergePolicyKind::Strict;
            return port;
        }

        static Port join(std::string_view name, MergePolicyKind policy = MergePolicyKind::Strict)
        {
            const auto trimmed = trim(name);
            if (trimmed.empty()) {
                throw std::invalid_argument("Port::join requires a non-empty name.");
            }

            auto [token, attribute] = split_token(trimmed);
            token = trim(token);
            attribute = trim(attribute);

            if (token.empty()) {
                throw std::invalid_argument("Join port specification cannot be empty.");
            }

            Port port{};
            port.kind = Kind::Join;
            port.identifier.assign(token.begin(), token.end());
            port.attribute.assign(attribute.begin(), attribute.end());
            port.merge_policy = policy;
            std::string repr{"join("};
            repr.append(trimmed.begin(), trimmed.end());
            repr.push_back(')');
            port.representation = std::move(repr);
            return port;
        }

        static Port join(
            std::initializer_list<std::string_view> names,
            MergePolicyKind policy = MergePolicyKind::Strict)
        {
            return join(names, policy, std::nullopt);
        }

        static Port join(
            std::initializer_list<std::string_view> names,
            MergePolicyKind policy,
            int64_t concat_dimension)
        {
            return join(names, policy, std::optional<int64_t>{concat_dimension});
        }

        static Port join(
            std::initializer_list<std::string_view> names,
            MergePolicyKind policy,
            std::optional<int64_t> concat_dimension)
        {
            if (names.size() == 0) {
                throw std::invalid_argument("Port::join requires at least one module name when using the aggregate form.");
            }

            std::vector<std::string> original_order;
            original_order.reserve(names.size());
            std::vector<std::string> canonical;
            canonical.reserve(names.size());

            for (auto name : names) {
                const auto trimmed = trim(name);
                if (trimmed.empty()) {
                    throw std::invalid_argument("Join specification contains an empty module name.");
                }
                original_order.emplace_back(trimmed.begin(), trimmed.end());
                canonical.push_back(original_order.back());
            }

            std::sort(canonical.begin(), canonical.end());
            canonical.erase(std::unique(canonical.begin(), canonical.end()), canonical.end());

            std::string identifier{"@join["};
            for (std::size_t index = 0; index < canonical.size(); ++index) {
                if (index > 0) {
                    identifier.append("|");
                }
                identifier.append(canonical[index]);
            }
            identifier.push_back(']');

            Port port{};
            port.kind = Kind::Join;
            port.identifier = std::move(identifier);
            port.merge_policy = policy;
            port.join_members = std::move(canonical);
            port.join_dimension = concat_dimension;

            if (concat_dimension) {
                port.attribute = "dim=" + std::to_string(*concat_dimension);
            }

            std::string repr{"join("};
            for (std::size_t index = 0; index < original_order.size(); ++index) {
                if (index > 0) {
                    repr.append(", ");
                }
                repr.append(original_order[index]);
            }
            if (concat_dimension) {
                repr.append("; dim=");
                repr.append(std::to_string(*concat_dimension));
            }
            repr.push_back(')');
            port.representation = std::move(repr);

            return port;
        }


    private:
        [[nodiscard]] std::string build_representation() const
        {
            if (!representation.empty()) {
                return representation;
            }
            std::string repr = identifier;
            if (!attribute.empty()) {
                repr.append(1, ':');
                repr.append(attribute);
            }
            return repr;
        }

        static std::pair<std::string_view, std::string_view> split_token(std::string_view token)
        {
            const auto position = token.find_first_of(".:");
            if (position == std::string_view::npos) {
                return {token, std::string_view{}};
            }
            return {token.substr(0, position), token.substr(position + 1)};
        }

        static std::string_view trim(std::string_view token)
        {
            while (!token.empty() && std::isspace(static_cast<unsigned char>(token.front()))) {
                token.remove_prefix(1);
            }
            while (!token.empty() && std::isspace(static_cast<unsigned char>(token.back()))) {
                token.remove_suffix(1);
            }
            return token;
        }
    };

    struct LinkSpec {
        Port source{};
        Port target{};

        LinkSpec() = default;
        LinkSpec(Port source, Port target) : source(std::move(source)), target(std::move(target)) {}
    };

    namespace Core {
        template <std::size_t BufferVRAMBatches>
        struct DevicePolicy {
            [[nodiscard]] static torch::Device select() {
                if constexpr (BufferVRAMBatches > 0) {
                    if (!torch::cuda::is_available()) {
                        throw std::runtime_error("CUDA device requested for VRAM buffering but is unavailable.");
                    }
                    return torch::Device(torch::kCUDA);
                } else {
                    return torch::Device(torch::kCPU);
                }
            }
        };

        template <std::size_t Epochs,
                std::size_t BatchSize,
                bool Shuffle,
                std::size_t BufferVRAMBatches,
                class DevicePolicyT = DevicePolicy<BufferVRAMBatches>>
        struct TrainingConfig {
            static_assert(Epochs > 0, "TrainingConfig requires at least one epoch.");
            static_assert(BatchSize > 0, "TrainingConfig requires a positive batch size.");

            static constexpr std::size_t epochs = Epochs;
            static constexpr std::size_t batch_size = BatchSize;
            static constexpr bool shuffle = Shuffle;
            static constexpr std::size_t buffer_vram = BufferVRAMBatches;

            using DevicePolicy = DevicePolicyT;
        };

        using SupervisedSample = std::pair<torch::Tensor, torch::Tensor>;
        using SupervisedDataset = std::vector<SupervisedSample>;

        using DefaultTrainingConfig = TrainingConfig<10, 32, true, 0>;
        inline constexpr auto kDefaultTrainingConfig = DefaultTrainingConfig{};
    }

    namespace ModelDetails {
        class SequentialBlockModuleImpl : public torch::nn::Module {
        public:
            explicit SequentialBlockModuleImpl(std::vector<Layer::Descriptor> layers)
            {
                std::size_t index{0};
                block_layers_.reserve(layers.size());
                for (auto& descriptor : layers) {
                    auto registered_layer = Layer::Details::build_registered_layer(*this, descriptor, index++);
                    block_layers_.push_back(std::move(registered_layer));
                }
            }

            torch::Tensor forward(torch::Tensor input)
            {
                auto output = std::move(input);
                for (auto& layer : block_layers_) {
                    output = layer.forward(std::move(output));
                    output = Activation::Details::apply(layer.activation, std::move(output));
                }
                return output;
            }

        private:
            std::vector<Layer::Details::RegisteredLayer> block_layers_{};
        };

        TORCH_MODULE(SequentialBlockModule);
    }


    // Execution mode for CUDA graph optimisation.
    enum class GraphMode {
        Disabled,
        Capture,
        Replay
    };

    struct TrainOptions {
        std::size_t epoch{Core::kDefaultTrainingConfig.epochs};
        std::size_t batch_size{Core::kDefaultTrainingConfig.batch_size};
        bool shuffle{Core::kDefaultTrainingConfig.shuffle};
        std::size_t buffer_vram{Core::kDefaultTrainingConfig.buffer_vram};
        bool monitor{true};
        bool restore_best_state{false};
        std::optional<std::pair<torch::Tensor, torch::Tensor>> validation{};
        std::optional<std::pair<torch::Tensor, torch::Tensor>> test{};
        std::ostream* stream{&std::cout};
        GraphMode graph_mode{GraphMode::Disabled};  // Enable CUDA graph capture/replay; pad or drop remainder batches first.
    };

    class Model : public torch::nn::Module {

        using RegularizationState = Regularization::StateVariant;
        using RegularizationStateStorage = std::shared_ptr<std::vector<RegularizationState>>;
        using RegularizationAccumulator = Regularization::Accumulator;
        using CalibrationMethod = Calibration::MethodPtr;

        struct RegularizationBinding {
            Regularization::Descriptor descriptor{};
            RegularizationStateStorage states{};
            RegularizationAccumulator accumulator{};
        };


    public:
        struct TrainingTelemetry {
            struct EpochSnapshot {
                std::size_t epoch_index{};
                double train_loss{};
                std::optional<double> test_loss{};
                std::optional<double> delta{};
                std::vector<double> learning_rates{};
                std::chrono::system_clock::time_point timestamp{};
                double duration_seconds{};
            };

            struct DatasetLossSnapshot {
                double loss{};
                std::size_t sample_count{};
                std::vector<double> learning_rates{};
                std::chrono::system_clock::time_point timestamp{};
            };

            [[nodiscard]] const std::vector<EpochSnapshot>& epochs() const noexcept { return epochs_; }
            [[nodiscard]] const std::vector<DatasetLossSnapshot>& dataset_losses() const noexcept { return dataset_losses_; }

            void clear() noexcept
            {
                epochs_.clear();
                dataset_losses_.clear();
            }

        private:
            friend class Model;

            void append_epoch(EpochSnapshot snapshot)
            {
                epochs_.push_back(std::move(snapshot));
            }

            void append_dataset_loss(DatasetLossSnapshot snapshot)
            {
                dataset_losses_.push_back(std::move(snapshot));
            }

            std::vector<EpochSnapshot> epochs_{};
            std::vector<DatasetLossSnapshot> dataset_losses_{};
        };

        explicit Model(std::string_view name = {}) : name_(name) {}
        [[nodiscard]] const TrainingTelemetry& training_telemetry() const noexcept { return telemetry_; }
        void clear_training_telemetry() noexcept { telemetry_.clear(); }
        using torch::nn::Module::train;
        [[nodiscard]] const std::string& name() const noexcept { return name_; }


        struct ForwardOptions {
            std::optional<std::size_t> max_chunk_size{};
            GraphMode graph_mode{GraphMode::Disabled};  // Graph capture/replay disables chunking; pad/drop to maintain static shapes.

            [[nodiscard]] bool buffering_enabled() const noexcept
            {
                return graph_mode == GraphMode::Disabled && max_chunk_size.has_value() && *max_chunk_size > 0;
            }

            [[nodiscard]] bool graph_capture_requested() const noexcept
            {
                return graph_mode == GraphMode::Capture;
            }

            [[nodiscard]] bool graph_replay_requested() const noexcept
            {
                return graph_mode == GraphMode::Replay;
            }
        };


        struct ModuleNameBinding {
            std::size_t entry{std::numeric_limits<std::size_t>::max()};
            std::size_t exit{std::numeric_limits<std::size_t>::max()};
            std::vector<std::size_t> layers{};

            [[nodiscard]] bool has_entry() const noexcept
            {
                return entry != std::numeric_limits<std::size_t>::max();
            }
        };

        struct CompiledNode {
            enum class Kind {
                Input,
                Module,
                Join,
                Output
            };

            Kind kind{Kind::Module};
            std::size_t index{std::numeric_limits<std::size_t>::max()};
            MergePolicyKind merge{MergePolicyKind::Strict};
            std::string label{};
            std::vector<std::size_t> inputs{};
            std::vector<std::size_t> outputs{};
        };

        struct CompiledStep {
            std::size_t node_index{std::numeric_limits<std::size_t>::max()};
            std::vector<std::size_t> dependencies{};
        };

        struct ExecutionStep {
            enum class Kind {
                Module,
                Join,
                Output
            };

            Kind kind{Kind::Module};
            std::size_t activation_index{std::numeric_limits<std::size_t>::max()};

            struct ModuleData {
                Layer::Details::RegisteredLayer* layer{nullptr};
                std::size_t input_index{std::numeric_limits<std::size_t>::max()};
            };

            struct JoinData {
                MergePolicyKind policy{MergePolicyKind::Strict};
                std::vector<std::size_t> producers{};
                std::size_t workspace_index{std::numeric_limits<std::size_t>::max()};
                std::optional<int64_t> concat_dimension{};
            };

            struct OutputData {
                std::size_t input_index{std::numeric_limits<std::size_t>::max()};
            };

            ModuleData module{};
            JoinData join{};
            OutputData output{};
        };

        struct JoinBuffer {
            std::size_t node_index{std::numeric_limits<std::size_t>::max()};
            MergePolicyKind policy{MergePolicyKind::Strict};
            std::vector<std::size_t> producers{};
            std::optional<int64_t> concat_dimension{};
        };

        struct GraphExecutionWorkspace {
            torch::Tensor input{};
            torch::Tensor output{};
            std::vector<torch::Tensor> node_buffers{};
            std::vector<std::vector<torch::Tensor>> join_scratch{};

            void invalidate() noexcept
            {
                input = torch::Tensor{};
                output = torch::Tensor{};
                node_buffers.clear();
                join_scratch.clear();
            }

            void ensure_node_capacity(std::size_t count)
            {
                if (node_buffers.size() != count) {
                    node_buffers.resize(count);
                }
            }

            void ensure_join_scratch(const std::vector<JoinBuffer>& join_buffers)
            {
                if (join_scratch.size() != join_buffers.size()) {
                    join_scratch.resize(join_buffers.size());
                }

                for (std::size_t index = 0; index < join_scratch.size(); ++index) {
                    auto& scratch = join_scratch[index];
                    scratch.reserve(join_buffers[index].producers.size());
                }
            }

            void bind_input(std::size_t index)
            {
                if (index >= node_buffers.size()) {
                    ensure_node_capacity(index + 1);
                }
                node_buffers[index] = input;
            }

            void bind_output(std::size_t index)
            {
                if (index >= node_buffers.size()) {
                    ensure_node_capacity(index + 1);
                }
                node_buffers[index] = output;
            }
        };




        using ModuleDescriptor = Common::SaveLoad::ModuleDescriptor;
        using NamedModuleDescriptor = Common::SaveLoad::NamedModuleDescriptor;
        void add(ModuleDescriptor descriptor, std::string name = {}) {
            if (regularization_configured_)
                throw std::logic_error("Cannot add modules after regularization has been configured.");

            clear_compiled_graph();

            const std::string module_name = std::move(name);
            if (!module_name.empty() && module_name_index_.find(module_name) != module_name_index_.end()) {
                throw std::invalid_argument("Module name '" + module_name + "' is already registered.");
            }

            ModuleDescriptor preserved_descriptor = descriptor;
            auto store_layer = [&](Layer::Details::RegisteredLayer registered_layer) {
                registered_layer.name = module_name;
                layers_.push_back(std::move(registered_layer));
                const auto layer_index = layers_.size() - 1;
                if (!module_name.empty()) {
                    auto& binding = module_name_index_[module_name];
                    if (!binding.has_entry()) {
                        binding.entry = layer_index;
                    }
                    binding.exit = layer_index;
                    binding.layers.push_back(layer_index);
                }
                register_layer_runtime(layers_.back());
            };
            auto register_layer = [&](auto&& concrete_descriptor) {
                using DescriptorType = std::decay_t<decltype(concrete_descriptor)>;
                auto registered = Layer::Details::build_registered_layer(
                    *this,
                    static_cast<const DescriptorType&>(concrete_descriptor),
                    next_module_index());
                store_layer(std::move(registered));
            };

            auto layer_dispatcher = Overloaded{
                [&](auto&& concrete_descriptor) {
                    register_layer(std::forward<decltype(concrete_descriptor)>(concrete_descriptor));
                }
            };

            auto sequential_block_handler = [&](Block::SequentialDescriptor sequential) {
                const bool block_declares_local_optimizer = sequential.local.optimizer.has_value();
                const bool any_layer_declares_local_optimizer = std::any_of(
                    sequential.layers.begin(),
                    sequential.layers.end(),
                    [](const Layer::Descriptor& layer_descriptor) {
                        return std::visit(
                            [](const auto& concrete_layer) {
                                return concrete_layer.local.optimizer.has_value();
                            }, layer_descriptor);
                    });

                if (block_declares_local_optimizer && !any_layer_declares_local_optimizer) {
                    const auto index = next_module_index();
                    auto module = register_module(
                        "sequential_block_" + std::to_string(index),
                        ModelDetails::SequentialBlockModule(std::move(sequential.layers)));

                    Layer::Details::RegisteredLayer registered_layer{};
                    registered_layer.activation = Activation::Type::Identity;
                    registered_layer.module = Layer::Details::to_shared_module_ptr(module);
                    registered_layer.local = std::move(sequential.local);
                    registered_layer.forward = [module](torch::Tensor input) {
                        return module->forward(std::move(input));
                    };

                    store_layer(std::move(registered_layer));
                } else {
                    if (block_declares_local_optimizer) {
                        for (auto& layer : sequential.layers) {
                            std::visit(
                                [&](auto& concrete_layer) {
                                    if (!concrete_layer.local.optimizer.has_value()) {
                                        concrete_layer.local = sequential.local;
                                    }
                                },
                                layer);
                        }

                    }
                    for (auto& layer : sequential.layers) {
                        std::visit(layer_dispatcher, std::move(layer));
                    }
                }
            };

            auto residual_block_handler = [&](Block::ResidualDescriptor residual) {
                auto residual_local = residual.local;
                const auto index = next_module_index();
                auto module = register_module(
                    "residual_block_" + std::to_string(index),
                    Block::Details::ResidualBlock(std::move(residual)));

                Layer::Details::RegisteredLayer registered_layer{};
                registered_layer.activation = Activation::Type::Identity;
                registered_layer.module = Layer::Details::to_shared_module_ptr(module);
                registered_layer.local = std::move(residual_local);
                registered_layer.forward = [module](torch::Tensor input) {
                    return module->forward(std::move(input));
                };

                store_layer(std::move(registered_layer));
            };

            auto transformer_block_handler = Overloaded{
                [&](Block::Transformer::Classic::EncoderDescriptor encoder_descriptor) {
                    const auto index = next_module_index();
                    auto module = register_module(
                        "transformer_encoder_" + std::to_string(index),
                        Block::Transformer::Classic::TransformerEncoder(std::move(encoder_descriptor)));

                    Layer::Details::RegisteredLayer registered_layer{};
                    registered_layer.activation = Activation::Type::Identity;
                    registered_layer.module = Layer::Details::to_shared_module_ptr(module);
                    registered_layer.forward = [module](torch::Tensor input) {
                        return module->forward(std::move(input));
                    };

                    store_layer(std::move(registered_layer));
                },
                [](const Block::Transformer::Classic::DecoderDescriptor&) {
                    throw std::invalid_argument("Transformer decoder blocks are not yet supported by Model::add.");
                },
            [&](Block::Transformer::EBT::EncoderDescriptor encoder_descriptor) {
                const auto index = next_module_index();
                auto module = register_module(
                    "ebt_encoder_" + std::to_string(index),
                    Block::Transformer::EBT::EncoderModule(std::move(encoder_descriptor)));

                    Layer::Details::RegisteredLayer registered_layer{};
                    registered_layer.activation = Activation::Type::Identity;
                    registered_layer.module = Layer::Details::to_shared_module_ptr(module);
                    registered_layer.forward = [module](torch::Tensor input) {
                        return module->forward(std::move(input));
                    };

                    store_layer(std::move(registered_layer));
                },
                [](const Block::Transformer::EBT::DecoderDescriptor&) {
                    throw std::invalid_argument("EBT decoder blocks are not yet supported by Model::add.");
                },
            [&](Block::Transformer::PlusPlus::EncoderDescriptor encoder_descriptor) {
                const auto index = next_module_index();
                auto module = register_module(
                    "transformer_pp_encoder_" + std::to_string(index),
                    Block::Transformer::PlusPlus::TransformerPlusPlusEncoder(std::move(encoder_descriptor)));

                Layer::Details::RegisteredLayer registered_layer{};
                registered_layer.activation = Activation::Type::Identity;
                registered_layer.module = Layer::Details::to_shared_module_ptr(module);
                registered_layer.forward = [module](torch::Tensor input) {
                    return module->forward(std::move(input));
                };

                layers_.push_back(std::move(registered_layer));
                register_layer_runtime(layers_.back());
                },
                [&](Block::Transformer::PlusPlus::DecoderDescriptor decoder_descriptor) {
                    const auto index = next_module_index();
                    auto module = register_module(
                        "transformer_pp_decoder_" + std::to_string(index),
                        Block::Transformer::PlusPlus::TransformerPlusPlusDecoder(std::move(decoder_descriptor)));

                    Layer::Details::RegisteredLayer registered_layer{};
                    registered_layer.activation = Activation::Type::Identity;
                    registered_layer.module = Layer::Details::to_shared_module_ptr(module);
                    registered_layer.forward = [module](torch::Tensor input) {
                        auto result = module->forward(std::move(input), torch::Tensor{});
                        return std::move(result.main);
                    };

                    store_layer(std::move(registered_layer));
                },
                [&](Block::Transformer::Mamba::EncoderDescriptor encoder_descriptor) {
                    const auto index = next_module_index();
                    auto module = register_module(
                        "mamba_encoder_" + std::to_string(index),
                        Block::Transformer::Mamba::EncoderModule(std::move(encoder_descriptor)));

                    Layer::Details::RegisteredLayer registered_layer{};
                    registered_layer.activation = Activation::Type::Identity;
                    registered_layer.module = Layer::Details::to_shared_module_ptr(module);
                    registered_layer.forward = [module](torch::Tensor input) {
                        return module->forward(std::move(input));
                    };

                    store_layer(std::move(registered_layer));
                }
            };

            auto block_dispatcher = Overloaded{
                [&](Block::SequentialDescriptor sequential) {
                    sequential_block_handler(std::move(sequential));
                },
                [&](Block::ResidualDescriptor residual) {
                    residual_block_handler(std::move(residual));
                },
                [&](Block::Transformer::Classic::EncoderDescriptor encoder_descriptor) {
                    transformer_block_handler(std::move(encoder_descriptor));
                },
                [&](Block::Transformer::Classic::DecoderDescriptor decoder_descriptor) {
                    transformer_block_handler(decoder_descriptor);
                },
                [&](Block::Transformer::EBT::EncoderDescriptor encoder_descriptor) {
                    transformer_block_handler(std::move(encoder_descriptor));
                },
                [&](Block::Transformer::EBT::DecoderDescriptor decoder_descriptor) {
                    transformer_block_handler(decoder_descriptor);
                },
                [&](Block::Transformer::PlusPlus::EncoderDescriptor encoder_descriptor) {
                    transformer_block_handler(std::move(encoder_descriptor));
                },
                [&](Block::Transformer::PlusPlus::DecoderDescriptor decoder_descriptor) {
                    transformer_block_handler(std::move(decoder_descriptor));
                },
                [&](Block::Transformer::Mamba::EncoderDescriptor encoder_descriptor) {
                    transformer_block_handler(std::move(encoder_descriptor));
                }
            };

            auto module_dispatcher = Overloaded{
                [&](Layer::Descriptor layer_descriptor) {
                    std::visit(layer_dispatcher, std::move(layer_descriptor));
                },
                [&](Block::Descriptor block_descriptor) {
                    std::visit(block_dispatcher, std::move(block_descriptor));
                }
            };

            std::visit(module_dispatcher, std::move(descriptor));
            module_descriptors_.emplace_back(std::move(preserved_descriptor), module_name);
        }

        void links(std::vector<LinkSpec> specifications)
        {
            if (specifications.empty()) {
                clear_compiled_graph();
                return;
            }

            std::vector<CompiledNode> nodes;
            std::vector<CompiledStep> steps;
            std::vector<JoinBuffer> joins;
            std::vector<LinkSpec> resolved_links;

            nodes.reserve(layers_.size() + specifications.size() * 2 + 2);
            resolved_links.reserve(specifications.size());

            CompiledNode input_node{};
            input_node.kind = CompiledNode::Kind::Input;
            input_node.label = "@input";
            nodes.push_back(std::move(input_node));
            const std::size_t input_node_index = 0;

            std::vector<std::size_t> module_node_indices(layers_.size(), std::numeric_limits<std::size_t>::max());
            for (std::size_t index = 0; index < layers_.size(); ++index) {
                CompiledNode node{};
                node.kind = CompiledNode::Kind::Module;
                node.index = index;
                const auto& layer = layers_[index];
                if (!layer.name.empty()) {
                    node.label = layer.name;
                    auto binding = module_name_index_.find(layer.name);
                    if (binding != module_name_index_.end() && binding->second.layers.size() > 1) {
                        const auto& sequence = binding->second.layers;
                        auto position = std::find(sequence.begin(), sequence.end(), index);
                        if (position != sequence.end()) {
                            node.label.append("[");
                            node.label.append(std::to_string(std::distance(sequence.begin(), position)));
                            node.label.append("]");
                        }
                    }
                } else {
                    node.label = "#" + std::to_string(index);
                }
                nodes.push_back(std::move(node));
                module_node_indices[index] = nodes.size() - 1;
            }

            std::optional<std::size_t> output_node_index{};
            std::unordered_map<std::string, std::size_t> join_lookup{};
            std::unordered_map<std::size_t, std::size_t> join_buffer_lookup{};

            auto parse_concat_dimension = [](const Port& port) -> std::optional<int64_t> {
                if (port.join_dimension) {
                    return port.join_dimension;
                }
                if (port.attribute.empty()) {
                    return std::nullopt;
                }

                std::string token;
                token.reserve(port.attribute.size());
                for (char ch : port.attribute) {
                    if (!std::isspace(static_cast<unsigned char>(ch))) {
                        token.push_back(ch);
                    }
                }

                auto strip_prefix = [](std::string& value, std::string_view prefix) {
                    if (value.rfind(prefix, 0) == 0) {
                        value.erase(0, prefix.size());
                    }
                };

                strip_prefix(token, "dim=");
                strip_prefix(token, "axis=");

                if (token.empty()) {
                    throw std::invalid_argument(
                        "Join port '" + port.describe() + "' specifies an empty concat dimension.");
                }

                try {
                    return std::stoll(token);
                } catch (const std::exception&) {
                    throw std::invalid_argument("Join port '" + port.describe() + "' specifies an invalid concat dimension '" + token + "'.");
                }
            };


            auto ensure_output_node = [&]() -> std::size_t {
                if (output_node_index) {
                    return *output_node_index;
                }
                CompiledNode node{};
                node.kind = CompiledNode::Kind::Output;
                node.label = "@output";
                nodes.push_back(std::move(node));
                output_node_index = nodes.size() - 1;
                return *output_node_index;
            };

            auto ensure_join_node = [&](const Port& port) -> std::size_t {
                const auto key = port.storage_key();
                auto it = join_lookup.find(key);
                if (it != join_lookup.end()) {
                    const auto node_index = it->second;
                    const auto buffer_index = join_buffer_lookup.at(node_index);
                    auto& buffer = joins[buffer_index];
                    if (buffer.policy != port.merge_policy) {
                        throw std::invalid_argument("Join node '" + port.describe()
                                                     + "' requested with conflicting merge policies.");
                    }
                    if (buffer.policy == MergePolicyKind::Concat) {
                        auto requested = parse_concat_dimension(port);
                        if (requested) {
                            if (buffer.concat_dimension && *requested != *buffer.concat_dimension) {
                                throw std::invalid_argument("Join node '" + port.describe()
                                                             + "' requested with conflicting concat dimensions.");
                            }
                            if (!buffer.concat_dimension) {
                                buffer.concat_dimension = requested;
                            }
                        }
                    }
                    return node_index;
                }

                CompiledNode node{};
                node.kind = CompiledNode::Kind::Join;
                node.merge = port.merge_policy;
                node.index = joins.size();
                node.label = port.describe();
                nodes.push_back(node);
                const std::size_t node_index = nodes.size() - 1;

                JoinBuffer buffer{};
                buffer.node_index = node_index;
                buffer.policy = port.merge_policy;
                if (buffer.policy == MergePolicyKind::Concat) {
                    buffer.concat_dimension = parse_concat_dimension(port);
                }
                join_buffer_lookup[node_index] = joins.size();
                joins.push_back(std::move(buffer));

                join_lookup.emplace(key, node_index);
                return node_index;
            };

            auto parse_numeric_identifier = [](std::string_view token) -> std::optional<std::size_t> {
                if (token.empty()) {
                    return std::nullopt;
                }
                if (token.front() == '#') {
                    token.remove_prefix(1);
                }
                if (token.empty()) {
                    return std::nullopt;
                }
                std::size_t value = 0;
                for (char character : token) {
                    if (!std::isdigit(static_cast<unsigned char>(character))) {
                        return std::nullopt;
                    }
                    value = value * 10 + static_cast<std::size_t>(character - '0');
                }
                return value;
            };

            enum class PortRole {
                Source,
                Target
            };

            auto resolve_module = [&](const Port& port, PortRole role) -> std::size_t {
                if (port.identifier.empty()) {
                    throw std::invalid_argument("Module port '" + port.describe() + "' is missing an identifier.");
                }

                std::optional<std::size_t> module_index{};
                auto by_name = module_name_index_.find(port.identifier);
                if (by_name != module_name_index_.end()) {
                    const auto& binding = by_name->second;
                    if (!binding.has_entry()) {
                        throw std::invalid_argument("Module port '" + port.describe() + "' references an unregistered module name.");
                    }
                    if (role == PortRole::Source) {
                        if (binding.exit == std::numeric_limits<std::size_t>::max()) {
                            throw std::invalid_argument("Module port '" + port.describe() + "' could not resolve an output endpoint.");
                        }
                        module_index = binding.exit;
                    } else {
                        if (binding.entry == std::numeric_limits<std::size_t>::max()) {
                            throw std::invalid_argument("Module port '" + port.describe() + "' could not resolve an input endpoint.");
                        }
                        module_index = binding.entry;
                    }
                } else {
                    module_index = parse_numeric_identifier(port.identifier);
                }

                if (!module_index.has_value() || *module_index >= module_node_indices.size()) {
                    throw std::invalid_argument("Unknown module referenced by port '" + port.describe() + "'.");
                }

                const auto node_index = module_node_indices[*module_index];
                if (node_index == std::numeric_limits<std::size_t>::max()) {
                    throw std::invalid_argument("Module port '" + port.describe() + "' could not be resolved.");
                }
                return node_index;
            };

            auto resolve_port = [&](Port& port, PortRole role) -> std::size_t {
                switch (port.kind) {
                    case Port::Kind::Input: {
                        port.assign_node(input_node_index);
                        return input_node_index;
                    }
                    case Port::Kind::Output: {
                        const auto index = ensure_output_node();
                        port.assign_node(index);
                        return index;
                    }
                    case Port::Kind::Module: {
                        const auto index = resolve_module(port, role);
                        port.assign_node(index);
                        return index;
                    }
                    case Port::Kind::Join: {
                        const auto index = ensure_join_node(port);
                        port.assign_node(index);
                        auto it = join_buffer_lookup.find(index);
                        if (it != join_buffer_lookup.end()) {
                            port.assign_join(it->second);
                        }
                        return index;
                    }
                }
                throw std::invalid_argument("Unsupported port kind encountered while resolving links.");
            };

            std::unordered_map<std::size_t, std::size_t> consumer_inbound{};

            std::unordered_set<std::string> auto_link_keys{};
            auto record_join_edge = [&](const LinkSpec& spec) {
                if (!spec.target.is_join()) {
                    return;
                }
                const auto key = spec.source.storage_key() + "->" + spec.target.storage_key();
                auto_link_keys.insert(key);
            };

            for (const auto& specification : specifications) {
                record_join_edge(specification);
            }

            std::vector<LinkSpec> inferred_links;
            inferred_links.reserve(specifications.size());
            auto schedule_join_members = [&](const Port& port) {
                if (!port.is_join() || port.join_members.empty()) {
                    return;
                }
                for (const auto& member : port.join_members) {
                    auto module_port = Port::parse(member);
                    auto join_port = port;
                    join_port.node_index.reset();
                    join_port.join_index.reset();
                    const auto key = module_port.storage_key() + "->" + join_port.storage_key();
                    if (auto_link_keys.insert(key).second) {
                        inferred_links.emplace_back(std::move(module_port), std::move(join_port));
                    }
                }
            };

            for (const auto& specification : specifications) {
                schedule_join_members(specification.source);
                schedule_join_members(specification.target);
            }

            specifications.insert(
                specifications.end(),
                std::make_move_iterator(inferred_links.begin()),
                std::make_move_iterator(inferred_links.end()));

            for (auto& specification : specifications) {
                auto link = specification;
                const auto source_index = resolve_port(link.source, PortRole::Source);
                const auto target_index = resolve_port(link.target, PortRole::Target);

                const auto source_kind = nodes[source_index].kind;
                const auto target_kind = nodes[target_index].kind;

                if (source_kind == CompiledNode::Kind::Output) {
                    throw std::invalid_argument("Output port '" + link.source.describe() + "' cannot be used as a source.");
                }

                if (target_kind == CompiledNode::Kind::Input) {
                    throw std::invalid_argument("Input port '" + link.target.describe() + "' cannot be used as a target.");
                }

                if (target_kind != CompiledNode::Kind::Join) {
                    auto& inbound = consumer_inbound[target_index];
                    if (inbound > 0) {
                        throw std::invalid_argument("Consumer port '" + link.target.describe()
                                                     + "' already has a producer.");
                    }
                    ++inbound;
                }

                nodes[source_index].outputs.push_back(target_index);
                nodes[target_index].inputs.push_back(source_index);

                if (target_kind == CompiledNode::Kind::Join) {
                    const auto buffer_index = join_buffer_lookup.at(target_index);
                    auto& buffer = joins[buffer_index];
                    if (std::find(buffer.producers.begin(), buffer.producers.end(), source_index)
                        != buffer.producers.end()) {
                        throw std::invalid_argument("Join node '" + link.target.describe()
                                                     + "' already receives input from '" + link.source.describe()
                                                     + "'.");
                    }
                    buffer.producers.push_back(source_index);
                }

                resolved_links.push_back(std::move(link));
            }

            for (const auto& [name, binding] : module_name_index_) {
                if (binding.layers.size() < 2) {
                    continue;
                }

                for (std::size_t offset = 1; offset < binding.layers.size(); ++offset) {
                    const auto upstream_layer = binding.layers[offset - 1];
                    const auto downstream_layer = binding.layers[offset];
                    if (upstream_layer >= module_node_indices.size() || downstream_layer >= module_node_indices.size()) {
                        throw std::invalid_argument("Module name '" + name + "' is out of sync with registered layers.");
                    }

                    const auto upstream_node = module_node_indices[upstream_layer];
                    const auto downstream_node = module_node_indices[downstream_layer];
                    if (upstream_node >= nodes.size() || downstream_node >= nodes.size()) {
                        throw std::invalid_argument("Module name '" + name + "' resolved to an invalid node index.");
                    }

                    auto& upstream_outputs = nodes[upstream_node].outputs;
                    if (std::find(upstream_outputs.begin(), upstream_outputs.end(), downstream_node) != upstream_outputs.end()) {
                        continue;
                    }

                    auto& downstream_inputs = nodes[downstream_node].inputs;
                    if (!downstream_inputs.empty()) {
                        throw std::invalid_argument("Module node '" + nodes[downstream_node].label
                                                     + "' already has a producer; unable to auto-link sequential block '"
                                                     + name + "'.");
                    }

                    auto& inbound = consumer_inbound[downstream_node];
                    if (inbound > 0) {
                        throw std::invalid_argument("Module node '" + nodes[downstream_node].label
                                                     + "' already has a producer; unable to auto-link sequential block '"
                                                     + name + "'.");
                    }
                    ++inbound;

                    upstream_outputs.push_back(downstream_node);
                    downstream_inputs.push_back(upstream_node);
                }
            }

            for (const auto& join : joins) {
                const auto& node = nodes[join.node_index];
                if (join.producers.empty()) {
                    throw std::invalid_argument("Join node '" + node.label + "' has no producers.");
                }
                if (node.outputs.empty()) {
                    throw std::invalid_argument("Join node '" + node.label + "' has no consumers.");
                }
            }

            for (const auto& node : nodes) {
                switch (node.kind) {
                    case CompiledNode::Kind::Input:
                        break;
                    case CompiledNode::Kind::Module:
                        if (node.inputs.empty()) {
                            throw std::invalid_argument(
                                "Module node '" + node.label + "' has no inbound links in the routing graph.");
                        }
                        break;
                    case CompiledNode::Kind::Join:
                        break;
                    case CompiledNode::Kind::Output:
                        if (node.inputs.empty()) {
                            throw std::invalid_argument("Output node has no inbound links in the routing graph.");
                        }
                        break;
                }
            }

            std::vector<std::size_t> indegree(nodes.size(), 0);
            for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
                for (auto target : nodes[node_index].outputs) {
                    if (target >= indegree.size()) {
                        throw std::invalid_argument("Link specification references an invalid node index.");
                    }
                    ++indegree[target];
                }
            }

            std::deque<std::size_t> queue;
            queue.clear();
            for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
                if (indegree[node_index] == 0) {
                    queue.push_back(node_index);
                }
            }

            steps.reserve(nodes.size());
            std::size_t visited = 0;
            while (!queue.empty()) {
                const auto node_index = queue.front();
                queue.pop_front();
                ++visited;

                if (node_index != input_node_index) {
                    CompiledStep step{};
                    step.node_index = node_index;
                    step.dependencies = nodes[node_index].inputs;
                    steps.push_back(std::move(step));
                }

                for (auto target : nodes[node_index].outputs) {
                    auto& degree = indegree[target];
                    if (degree == 0) {
                        continue;
                    }
                    --degree;
                    if (degree == 0) {
                        queue.push_back(target);
                    }
                }
            }

            if (visited != nodes.size()) {
                throw std::invalid_argument("Link specification contains cycles; unable to compile routing graph.");
            }

            std::vector<ExecutionStep> execution_steps;
            execution_steps.reserve(steps.size());

            for (const auto& step : steps) {
                const auto node_index = step.node_index;
                const auto& node = nodes[node_index];

                ExecutionStep execution{};
                execution.activation_index = node_index;

                switch (node.kind) {
                    case CompiledNode::Kind::Input: {
                        continue;
                    }
                    case CompiledNode::Kind::Module: {
                        if (node.index >= layers_.size()) {
                            throw std::invalid_argument(
                                "Module node '" + node.label + "' references an invalid layer index.");
                        }
                        if (step.dependencies.size() != 1) {
                            throw std::invalid_argument(
                                "Module node '" + node.label + "' must have exactly one dependency.");
                        }

                        execution.kind = ExecutionStep::Kind::Module;
                        execution.module.layer = &layers_[node.index];
                        execution.module.input_index = step.dependencies.front();
                        break;
                    }
                    case CompiledNode::Kind::Join: {
                        if (node.index >= joins.size()) {
                            throw std::invalid_argument(
                                "Join node '" + node.label + "' references an invalid join buffer index.");
                        }
                        const auto& buffer = joins[node.index];
                        if (buffer.producers.empty()) {
                            throw std::invalid_argument(
                                "Join node '" + node.label + "' has no producers after compilation.");
                        }

                        execution.kind = ExecutionStep::Kind::Join;
                        execution.join.policy = buffer.policy;
                        execution.join.producers = buffer.producers;
                        execution.join.workspace_index = node.index;
                        execution.join.concat_dimension = buffer.concat_dimension;
                        break;
                    }
                    case CompiledNode::Kind::Output: {
                        if (step.dependencies.size() != 1) {
                            throw std::invalid_argument("Output node must have exactly one dependency.");
                        }

                        execution.kind = ExecutionStep::Kind::Output;
                        execution.output.input_index = step.dependencies.front();
                        break;
                    }
                }

                execution_steps.push_back(std::move(execution));
            }

            const std::size_t resolved_output_index = output_node_index.value_or(
                steps.empty() ? input_node_index : steps.back().node_index);
            if (resolved_output_index >= nodes.size()) {
                throw std::invalid_argument("Resolved output node index is out of range for the compiled graph.");
            }

            compiled_nodes_ = std::move(nodes);
            compiled_steps_ = std::move(steps);
            join_buffers_ = std::move(joins);
            compiled_links_ = std::move(resolved_links);
            compiled_output_node_index_ = output_node_index;
            routing_active_ = true;
            invalidate_execution_workspace();
        }

        [[nodiscard]] bool has_compiled_routing() const noexcept { return routing_active_; }
        [[nodiscard]] const std::vector<CompiledNode>& compiled_nodes() const noexcept { return compiled_nodes_; }
        [[nodiscard]] const std::vector<CompiledStep>& compiled_steps() const noexcept { return compiled_steps_; }
        [[nodiscard]] const std::vector<JoinBuffer>& join_buffers() const noexcept { return join_buffers_; }
        [[nodiscard]] const std::vector<LinkSpec>& compiled_links() const noexcept { return compiled_links_; }


        void set_optimizer(Optimizer::Descriptor descriptor, std::optional<LrScheduler::Descriptor> scheduler = std::nullopt) {
            if (layers_.empty()) {
                throw std::logic_error("Cannot create optimizer before any layer has been registered.");
            }
            auto build_optimizer_for = [](const Optimizer::Descriptor& config, std::vector<torch::Tensor> parameters) {
                return std::visit(
                    [&](const auto& concrete_descriptor) -> std::unique_ptr<torch::optim::Optimizer> {
                        using DescriptorType = std::decay_t<decltype(concrete_descriptor)>;
                        if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::SGDDescriptor>) {
                            auto options = Optimizer::Details::to_torch_options(concrete_descriptor.options);
                            return std::make_unique<torch::optim::SGD>(std::move(parameters), options);
                        } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::AdamWDescriptor>) {
                            auto options = Optimizer::Details::to_torch_options(concrete_descriptor.options);
                            return std::make_unique<torch::optim::AdamW>(std::move(parameters), options);
                        } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::SophiaGDescriptor>) {
                            return std::make_unique<Optimizer::Details::SophiaG>(std::move(parameters), concrete_descriptor.options);
                        } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::SophiaHDescriptor>) {
                            return std::make_unique<Optimizer::Details::SophiaH>(std::move(parameters), concrete_descriptor.options);
                        } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::MuonDescriptor>) {
                            return std::make_unique<Optimizer::Details::Muon>(std::move(parameters), concrete_descriptor.options);
                        } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::AdaMuonDescriptor>) {
                            return std::make_unique<Optimizer::Details::AdaMuon>(std::move(parameters), concrete_descriptor.options);
                        } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::MuonManifoldDescriptor>) {
                            return std::make_unique<Optimizer::Details::MuonManifold>(std::move(parameters), concrete_descriptor.options);
                        } else {
                            static_assert(sizeof(DescriptorType) == 0, "Unsupported optimizer descriptor provided to Model::set_optimizer.");
                        }
                    }, config);
            };


            optimizer_.reset();
            local_optimizers_.clear();

            std::vector<torch::Tensor> global_parameters{};
            for (const auto& layer : layers_) {
                if (!layer.module) {
                    if (layer.local.optimizer.has_value()) {
                        throw std::logic_error("Local optimizer requested for a layer without a registered module.");
                    }
                    continue;
                }

                auto parameters = [&]() {
                    if (auto sequential_block = std::dynamic_pointer_cast<ModelDetails::SequentialBlockModuleImpl>(layer.module)) {
                        return sequential_block->parameters();
                    }
                    return layer.module->parameters();
                }();
                if (parameters.empty()) {
                    if (layer.local.optimizer.has_value()) {
                        throw std::logic_error("Local optimizer requested for a layer without trainable parameters.");
                    }
                    continue;
                }

                if (layer.local.optimizer.has_value()) {
                    local_optimizers_.push_back(build_optimizer_for(*layer.local.optimizer, std::move(parameters)));
                } else {
                    global_parameters.insert(global_parameters.end(), parameters.begin(), parameters.end());
                }
            }

            if (!global_parameters.empty()) {
                optimizer_ = build_optimizer_for(descriptor, std::move(global_parameters));
            }

            scheduler_.reset();
            if (scheduler.has_value()) {
                if (!optimizer_)
                    throw std::logic_error("Cannot attach a scheduler without a global optimizer.");
                scheduler_ = std::visit(
                    [&](const auto& concrete_descriptor) -> std::unique_ptr<LrScheduler::Details::Scheduler> {
                        return LrScheduler::Details::build_scheduler(*this, *optimizer_, concrete_descriptor);
                    }, std::move(*scheduler));

            }
            configure_step_impl();

        }

        template <class Descriptor>
        void set_loss(Descriptor descriptor) {
            using Decayed = std::decay_t<Descriptor>;
            constexpr bool kSupported = std::disjunction_v<
                std::is_same<Decayed, Loss::MSEDescriptor>,
                std::is_same<Decayed, Loss::CrossEntropyDescriptor>>;
            static_assert(kSupported, "Unsupported loss descriptor type provided to Model::set_loss.");

            loss_descriptor_ = LossDescriptor{std::in_place_type<Decayed>, std::move(descriptor)};
        }

        void set_regularization(std::vector<Regularization::Descriptor> descriptors)
        {
            if (regularization_configured_)
                throw std::logic_error("Regularization descriptors have already been configured.");
            regularization_configured_ = true;
            global_regularization_parameters_ = collect_global_trainable_parameters();
            global_regularization_bindings_.clear();
            global_regularization_bindings_.reserve(descriptors.size());

            for (auto& descriptor : descriptors) {
                global_regularization_bindings_.push_back(
                    make_regularization_binding(std::move(descriptor), global_regularization_parameters_));
            }
        }

        void clear_regularization() noexcept {
            for (auto& bindings : layer_regularization_bindings_) {
                bindings.clear();
            }
            global_regularization_bindings_.clear();
            global_regularization_parameters_.clear();
            regularization_configured_ = false;
        }

        [[nodiscard]] bool has_regularization() const noexcept {
            if (!global_regularization_bindings_.empty())
                return true;
            return std::any_of(
                layer_regularization_bindings_.begin(),
                layer_regularization_bindings_.end(),
                [](const auto& bindings) { return !bindings.empty(); });
        }

        [[nodiscard]] torch::Tensor compute_regularization_penalty() const {
            const auto fallback_options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
            const auto fallback = std::optional<torch::TensorOptions>{fallback_options};



            torch::Tensor total;
            bool initialised = false;

            auto accumulate_penalty = [&](const RegularizationBinding& binding, const std::vector<torch::Tensor>& parameters) {
                if (!binding.accumulator) {
                    return;
                }

                auto penalty = binding.accumulator(parameters, fallback);
                if (!penalty.defined()) {
                    return;
                }


                if (!initialised) {
                    total = penalty;
                    initialised = true;
                    return;
                }
                if (penalty.device() != total.device()) {
                    penalty = penalty.to(total.device());
                }
                if (penalty.scalar_type() != total.scalar_type()) {
                    penalty = penalty.to(total.scalar_type());
                }
                total.add_(penalty);
            };

            for (const auto& binding : global_regularization_bindings_) {
                accumulate_penalty(binding, global_regularization_parameters_);
            }

            for (std::size_t index = 0; index < layer_regularization_bindings_.size(); ++index) {
                const auto& bindings = layer_regularization_bindings_[index];
                if (bindings.empty()) {
                    continue;
                }

                const auto& parameters = layer_parameters_[index];
                for (const auto& binding : bindings) {
                    accumulate_penalty(binding, parameters);
                }
            }

            if (!initialised) {
                return torch::zeros({}, fallback_options);
            }

            return total;
        }

        Model& to_device(bool use_cuda = true)
        {
            if (use_cuda) {
                if (!torch::cuda::is_available()) {
                    throw std::runtime_error("CUDA device requested but is unavailable.");
                }
                device_ = torch::Device(torch::kCUDA, /*index=*/0);
            } else {
                device_ = torch::Device(torch::kCPU, /*index=*/0);
            }

            this->to(device_);
            return *this;
        }

        [[nodiscard]] const torch::Device& device() const noexcept { return device_; }


        [[nodiscard]] torch::Tensor forward(torch::Tensor input)
        {
            return forward(std::move(input), {});
        }

        [[nodiscard]] torch::Tensor forward(torch::Tensor input, ForwardOptions options)
        {
            auto execute = [&](torch::Tensor tensor) {
                return execute_plan(std::move(tensor));
            };

            if (input.defined() && input.device() != device_) {
                input = input.to(device_);
            }

            const auto graph_mode = options.graph_mode;
            const bool graph_mode_active = graph_mode != GraphMode::Disabled;
            if (graph_mode_active) {
                ensure_graph_input_shape(graph_mode, input);
            }


            const bool can_buffer = options.buffering_enabled() && input.defined() && input.dim() > 0;
            if (!can_buffer) {
                return execute(std::move(input));
            }

            const auto chunk_limit = static_cast<int64_t>(*options.max_chunk_size);
            if (chunk_limit <= 0) {
                return execute(std::move(input));
            }
            const auto leading = input.size(0);
            if (leading == 0 || leading <= chunk_limit) {
                return execute(std::move(input));
            }
            std::vector<torch::Tensor> outputs;
            outputs.reserve(static_cast<std::size_t>((leading + chunk_limit - 1) / chunk_limit));

            for (int64_t offset = 0; offset < leading; offset += chunk_limit) {
                const auto current = std::min<int64_t>(chunk_limit, leading - offset);
                auto chunk = input.narrow(0, offset, current);
                outputs.push_back(execute(std::move(chunk)));
            }

            return torch::cat(outputs, 0);
        }

        torch::Tensor execute_plan(torch::Tensor tensor)
        {
            if (tensor.defined() && tensor.device() != device_) {
                tensor = tensor.to(device_);
            }

            auto apply_calibrations = [&](torch::Tensor value) {
                for (const auto& calibration : calibration_methods_) {
                    value = calibration->transform(std::move(value));
                }
                return value;
            };

            if (!has_compiled_routing() || compiled_nodes_.empty()) {
                for (auto& layer : layers_) {
                    tensor = layer.forward(std::move(tensor));
                    tensor = Activation::Details::apply(layer.activation, std::move(tensor));
                }
                return apply_calibrations(std::move(tensor));
            }

            ensure_execution_workspace();

            constexpr std::size_t kInputNodeIndex = 0;
            copy_into_graph_input_buffer(std::move(tensor));

            auto& workspace = graph_workspace_;


            const auto output_index = compiled_output_node_index_.value_or(compiled_steps_.empty() ? kInputNodeIndex : compiled_steps_.back().node_index);
            workspace.bind_output(output_index);

            for (const auto& step : compiled_steps_) {
                const auto node_index = step.node_index;
                if (node_index >= compiled_nodes_.size()) {
                    throw std::runtime_error("Compiled step references an unknown node index.");
                }

                const auto& node = compiled_nodes_[node_index];
                switch (node.kind) {
                    case CompiledNode::Kind::Input: {
                        break;
                    }
                    case CompiledNode::Kind::Module: {
                        if (node.index >= cached_layer_pointers_.size()) {
                            throw std::runtime_error("Module cache is out of sync with compiled graph.");
                        }
                        if (step.dependencies.empty()) {
                            throw std::runtime_error("Module node is missing its dependency in the compiled graph.");
                        }
                        const auto upstream_index = step.dependencies.front();
                        if (upstream_index >= workspace.node_buffers.size()) {
                            throw std::runtime_error("Module dependency index is invalid.");
                        }
                        auto input_tensor = workspace.node_buffers[upstream_index];
                        if (!input_tensor.defined()) {
                            throw std::runtime_error("Module dependency tensor is undefined during plan execution.");
                        }
                        auto* layer = cached_layer_pointers_[node.index];
                        if (!layer) {
                            throw std::runtime_error("Cached layer pointer is null during plan execution.");
                        }
                        auto output_tensor = layer->forward(input_tensor);
                        output_tensor = Activation::Details::apply(layer->activation, std::move(output_tensor));
                        auto& destination = workspace.node_buffers[node_index];
                        copy_tensor_into(destination, output_tensor);
                        break;
                    }
                    case CompiledNode::Kind::Join: {
                        if (node.index >= join_buffers_.size()) {
                            throw std::runtime_error("Join buffer cache is out of sync with compiled graph.");
                        }
                        auto& buffer = join_buffers_[node.index];
                        auto& scratch = workspace.join_scratch[node.index];
                        scratch.clear();
                        scratch.reserve(buffer.producers.size());
                        for (auto producer : buffer.producers) {
                            if (producer >= workspace.node_buffers.size()) {
                                throw std::runtime_error("Join producer index is invalid.");
                            }
                            auto value = workspace.node_buffers[producer];
                            if (!value.defined()) {
                                throw std::runtime_error("Join producer tensor is undefined during plan execution.");
                            }
                            scratch.push_back(value);
                        }

                        torch::Tensor joined;
                        switch (buffer.policy) {
                            case MergePolicyKind::Strict: {
                                if (scratch.size() != 1) {
                                    throw std::runtime_error("Strict join expected exactly one producer.");
                                }
                                joined = scratch.front();
                                break;
                            }
                            case MergePolicyKind::Broadcast: {
                                if (scratch.empty()) {
                                    throw std::runtime_error("Broadcast join has no producers.");
                                }
                                joined = scratch.front();
                                for (std::size_t index = 1; index < scratch.size(); ++index) {
                                    joined = joined + scratch[index];
                                }
                                break;
                            }
                            case MergePolicyKind::Concat: {
                                if (scratch.empty()) {
                                    throw std::runtime_error("Concat join has no producers.");
                                }
                                const auto dimension = buffer.concat_dimension.value_or(1);
                                joined = torch::cat(scratch, dimension);
                                break;
                            }
                        }

                        auto& destination = workspace.node_buffers[node_index];
                        copy_tensor_into(destination, joined);

                        scratch.clear();
                        break;
                    }
                    case CompiledNode::Kind::Output: {
                        if (step.dependencies.empty()) {
                            throw std::runtime_error("Output node has no dependencies in the compiled graph.");
                        }
                        const auto upstream_index = step.dependencies.front();
                        if (upstream_index >= workspace.node_buffers.size()) {
                            throw std::runtime_error("Output dependency index is invalid.");
                        }
                        auto upstream_tensor = workspace.node_buffers[upstream_index];
                        if (!upstream_tensor.defined()) {
                            throw std::runtime_error("Output dependency tensor is undefined during plan execution.");
                        }
                        copy_tensor_into(workspace.output, upstream_tensor);
                        workspace.bind_output(node_index);
                        break;
                    }
                }
            }

            if (output_index >= workspace.node_buffers.size()) {
                throw std::runtime_error("Compiled output index is invalid.");
            }

            auto output_tensor = workspace.node_buffers[output_index];
            if (!output_tensor.defined()) {
                throw std::runtime_error("Model::forward produced an undefined tensor at the output node.");
            }

            copy_tensor_into(workspace.output, output_tensor);
            workspace.bind_output(output_index);

            auto result = graph_output_tensor();

            return apply_calibrations(std::move(result));
        }


        void set_model_name(std::string name) { model_name_ = std::move(name); }

        [[nodiscard]] std::string model_name() const
        {
            if (model_name_.empty()) {
                return this->name();
            }
            return model_name_;
        }

        void save(const std::filesystem::path& directory) const
        {
            namespace fs = std::filesystem;
            if (directory.empty()) {
                throw std::invalid_argument("Model::save requires a non-empty directory path.");
            }

            fs::create_directories(directory);

            const auto architecture_path = directory / "architecture.json";
            const auto parameters_path = directory / "parameters.binary";

            Common::SaveLoad::PropertyTree architecture;
            architecture.put("name", model_name());
            architecture.add_child("modules", Common::SaveLoad::serialize_module_list(module_descriptors_));

            try {
                Common::SaveLoad::write_json_file(architecture_path, architecture);
            } catch (const std::exception& error) {
                throw std::runtime_error(std::string("Failed to write architecture description to '")
                                         + architecture_path.string() + "': " + error.what());
            }

            torch::serialize::OutputArchive archive;
            torch::nn::Module::save(archive);
            archive.save_to(parameters_path.string());
        }

        void load(const std::filesystem::path& directory)
        {
            namespace fs = std::filesystem;
            if (directory.empty()) {
                throw std::invalid_argument("Model::load requires a non-empty directory path.");
            }

            const auto architecture_path = directory / "architecture.json";
            const auto parameters_path = directory / "parameters.binary";

            if (!fs::exists(architecture_path)) {
                throw std::runtime_error(std::string("Architecture file not found at '")
                                         + architecture_path.string() + "'.");
            }
            if (!fs::exists(parameters_path)) {
                throw std::runtime_error(std::string("Parameter archive not found at '")
                                         + parameters_path.string() + "'.");
            }

            Common::SaveLoad::PropertyTree architecture;
            try {
                architecture = Common::SaveLoad::read_json_file(architecture_path);
            } catch (const std::exception& error) {
                throw std::runtime_error(std::string("Failed to read architecture description from '")
                                         + architecture_path.string() + "': " + error.what());
            }

            auto modules_node = architecture.get_child_optional("modules");
            if (!modules_node) {
                throw std::runtime_error(std::string("Architecture description '") + architecture_path.string()
                                         + "' is missing the 'modules' entry.");
            }

            auto descriptors = Common::SaveLoad::deserialize_module_list(*modules_node, "module");

            reset_runtime_state();

            if (auto name_value = architecture.get_optional<std::string>("name")) {
                model_name_ = std::move(*name_value);
            } else {
                model_name_.clear();
            }

            for (auto& descriptor : descriptors) {
                add(std::move(descriptor.descriptor), std::move(descriptor.name));
            }

            torch::serialize::InputArchive validation_archive;
            try {
                validation_archive.load_from(parameters_path.string());
            } catch (const c10::Error& error) {
                throw std::runtime_error(std::string("Failed to open parameter archive '")
                                         + parameters_path.string() + "': " + error.what());
            }

            auto parameters = this->named_parameters(/*recurse=*/true);
            for (const auto& item : parameters) {
                torch::Tensor stored;
                try {
                    validation_archive.read(item.key(), stored);
                } catch (const c10::Error& error) {
                    throw std::runtime_error("Checkpoint is missing parameter '" + item.key() + "': " + error.what());
                }
                if (!stored.defined()) {
                    throw std::runtime_error("Checkpoint parameter '" + item.key() + "' is undefined.");
                }
                if (stored.sizes() != item.value().sizes()) {
                    throw std::runtime_error("Parameter '" + item.key() + "' shape mismatch: expected "
                                             + format_tensor_shape(item.value()) + " but found "
                                             + format_tensor_shape(stored) + ".");
                }
            }

            auto buffers = this->named_buffers(/*recurse=*/true);
            for (const auto& item : buffers) {
                if (!item.value().defined()) {
                    continue;
                }
                torch::Tensor stored;
                try {
                    validation_archive.read(item.key(), stored);
                } catch (const c10::Error& error) {
                    throw std::runtime_error("Checkpoint is missing buffer '" + item.key() + "': " + error.what());
                }
                if (!stored.defined()) {
                    throw std::runtime_error("Checkpoint buffer '" + item.key() + "' is undefined.");
                }
                if (stored.sizes() != item.value().sizes()) {
                    throw std::runtime_error("Buffer '" + item.key() + "' shape mismatch: expected "
                                             + format_tensor_shape(item.value()) + " but found "
                                             + format_tensor_shape(stored) + ".");
                }
            }

            torch::serialize::InputArchive archive;
            try {
                archive.load_from(parameters_path.string());
                torch::nn::Module::load(archive);
            } catch (const c10::Error& error) {
                throw std::runtime_error(std::string("Failed to load parameters from '")
                                         + parameters_path.string() + "': " + error.what());
            }

            configure_step_impl();
        }

        void calibrate(const torch::Tensor& inputs, const torch::Tensor& targets, const Calibration::Descriptor& descriptor,  bool plot = true,
                       std::optional<std::pair<torch::Tensor, torch::Tensor>> validation = std::nullopt, Calibration::Options options = {}) {
            torch::NoGradGuard guard;
            eval();
            if (!inputs.defined() || !targets.defined()) {
                throw std::invalid_argument("Calibration requires defined input and target tensors.");
            }

            auto maybe_chunked_forward = [this, &options](const torch::Tensor& dataset_inputs, const torch::Tensor& dataset_targets) -> std::optional<std::pair<torch::Tensor, torch::Tensor>> {
                const bool buffering_enabled = options.forward_buffer_batches > 0;
                const bool has_leading_dimension = dataset_inputs.dim() > 0;
                if (!buffering_enabled || !has_leading_dimension || dataset_inputs.size(0) <= 0) {
                    return std::nullopt;
                }



                const auto chunk_size_value = static_cast<std::int64_t>(options.forward_chunk_size.value_or(Core::kDefaultTrainingConfig.batch_size));
                if (chunk_size_value <= 0) {
                    throw std::invalid_argument("Calibration forward chunk size must be positive when buffering is enabled.");
                }

                const auto total_samples = dataset_inputs.size(0);
                const auto total_batches = static_cast<std::size_t>((total_samples + chunk_size_value - 1) / chunk_size_value);

                if (total_batches == 0) {
                    return std::nullopt;
                }

                std::deque<std::pair<torch::Tensor, torch::Tensor>> buffered_batches;
                std::size_t next_batch_to_load = 0;
                const std::size_t buffer_limit = std::max<std::size_t>(1, options.forward_buffer_batches);

                auto fetch_batch = [&](std::size_t batch_index) {
                    const auto offset = static_cast<std::int64_t>(batch_index) * chunk_size_value;
                    const auto remaining = total_samples - offset;
                    const auto current_batch = std::min<std::int64_t>(chunk_size_value, remaining);
                    if (current_batch <= 0) {
                        return std::pair<torch::Tensor, torch::Tensor>{torch::Tensor{}, torch::Tensor{}};
                    }

                    auto batch_inputs = dataset_inputs.narrow(0, offset, current_batch);
                    if (batch_inputs.device() != device_) {
                        batch_inputs = batch_inputs.to(device_);
                    }

                    auto batch_targets = dataset_targets.narrow(0, offset, current_batch);
                    if (!batch_targets.device().is_cpu()) {
                        batch_targets = batch_targets.to(torch::kCPU);
                    }

                    return std::pair<torch::Tensor, torch::Tensor>{std::move(batch_inputs), std::move(batch_targets)};
                };

                auto maintain_buffer = [&](std::size_t current_index) {
                    const auto desired_size = std::min(buffer_limit, total_batches - current_index);
                    while (buffered_batches.size() < desired_size && next_batch_to_load < total_batches) {
                        auto batch = fetch_batch(next_batch_to_load);
                        if (batch.first.defined() && batch.second.defined()) {
                            buffered_batches.push_back(std::move(batch));
                        }
                        ++next_batch_to_load;
                    }
                };

                std::vector<torch::Tensor> logits_chunks;
                logits_chunks.reserve(total_batches);
                std::vector<torch::Tensor> target_chunks;
                target_chunks.reserve(total_batches);

                for (std::size_t batch_index = 0; batch_index < total_batches; ++batch_index) {
                    maintain_buffer(batch_index);
                    if (buffered_batches.empty()) {
                        break;
                    }

                    auto batch_pair = std::move(buffered_batches.front());
                    buffered_batches.pop_front();

                    auto batch_logits = forward(std::move(batch_pair.first));
                    if (!batch_logits.device().is_cpu()) {
                        batch_logits = batch_logits.to(torch::kCPU);
                    }
                    logits_chunks.push_back(batch_logits.detach());

                    auto batch_targets = std::move(batch_pair.second);
                    if (!batch_targets.device().is_cpu()) {
                        batch_targets = batch_targets.to(torch::kCPU);
                    }
                    target_chunks.push_back(std::move(batch_targets));

                    maintain_buffer(batch_index + 1);
                }

                if (!logits_chunks.empty() && !target_chunks.empty()) {
                    return std::pair<torch::Tensor, torch::Tensor>{
                        torch::cat(logits_chunks, 0),
                        torch::cat(target_chunks, 0)
                    };
                }

                return std::nullopt;
            };

            auto calibration_pair = maybe_chunked_forward(inputs, targets);
            torch::Tensor logits;
            torch::Tensor calibration_targets;

            if (calibration_pair.has_value()) {
                logits = std::move(calibration_pair->first);
                calibration_targets = std::move(calibration_pair->second);
            } else {
                ForwardOptions forward_options{};
                if (options.forward_chunk_size.has_value()) {
                    forward_options.max_chunk_size = options.forward_chunk_size;
                }
                auto fallback_logits = forward(inputs, std::move(forward_options));
                if (!fallback_logits.device().is_cpu()) {
                    fallback_logits = fallback_logits.to(torch::kCPU);
                }
                logits = fallback_logits.detach();
                calibration_targets = targets;
                if (!calibration_targets.device().is_cpu()) {
                    calibration_targets = calibration_targets.to(torch::kCPU);
                }
            }

            std::optional<std::pair<torch::Tensor, torch::Tensor>> processed_validation = std::nullopt;
            if (validation.has_value()) {
                const auto& validation_inputs = validation->first;
                const auto& validation_targets = validation->second;
                if (validation_inputs.defined() && validation_targets.defined()) {
                    if (auto validation_pair = maybe_chunked_forward(validation_inputs, validation_targets)) {
                        processed_validation = std::move(*validation_pair);
                    } else {
                        ForwardOptions forward_options{};
                        if (options.forward_chunk_size.has_value()) {
                            forward_options.max_chunk_size = options.forward_chunk_size;
                        }
                        auto validation_logits = forward(validation_inputs, std::move(forward_options));
                        if (!validation_logits.device().is_cpu()) {
                            validation_logits = validation_logits.to(torch::kCPU);
                        }
                        auto validation_targets_cpu = validation_targets;
                        if (!validation_targets_cpu.device().is_cpu()) {
                            validation_targets_cpu = validation_targets_cpu.to(torch::kCPU);
                        }
                        processed_validation = std::make_pair(validation_logits.detach(), validation_targets_cpu);
                    }
                }
            }

            auto method = Calibration::Calibrate(*this, device_, descriptor, [&logits, &calibration_targets](torch::nn::Module&) {
                return std::pair<torch::Tensor, torch::Tensor>{logits, calibration_targets};
            }, std::move(processed_validation), std::move(options), plot);
            calibration_methods_.push_back(std::move(method));
        }

        [[nodiscard]] torch::Tensor compute_loss(const torch::Tensor& prediction,
                                                 const torch::Tensor& target,
                                                 const std::optional<torch::Tensor>& weight = std::nullopt) const {
            if (!loss_descriptor_.has_value()) {
                throw std::logic_error("Loss function has not been configured.");
            }
            return std::visit(
                            [&](const auto& descriptor) {
                                return Loss::Details::compute(descriptor, prediction, target, weight);
                            }, *loss_descriptor_);
        }

        auto evaluate(torch::Tensor evaluation_inputs, torch::Tensor evaluation_targets, Evaluation::ClassificationDescriptor descriptor, std::vector<Metric::Classification::Descriptor> metrics, Evaluation::Options options = {}) -> Evaluation::ClassificationReport {
            return Evaluation::Evaluate(
                *this,
                std::move(evaluation_inputs),
                std::move(evaluation_targets),
                descriptor,
                std::move(metrics),
                options);
        }

        template <class Descriptor, class... Args>
        decltype(auto) plot(Descriptor descriptor, Args&&... args);

        void zero_grad(bool set_to_none = false) {
            bool handled{false};

            if (optimizer_) {
                optimizer_->zero_grad(set_to_none);
                handled = true;
            }
            if (!local_optimizers_.empty()) {
                handled = true;
                for (auto& optimizer : local_optimizers_) {
                    optimizer->zero_grad(set_to_none);
                }
            }

            if (!handled) {
                torch::nn::Module::zero_grad(set_to_none);
            }
        }

        void step() { (this->*step_impl_)(); }

        [[nodiscard]] bool has_optimizer() const noexcept {
            return static_cast<bool>(optimizer_) || !local_optimizers_.empty();
        }
        [[nodiscard]] std::size_t local_optimizer_count() const noexcept {
            return local_optimizers_.size();
        }

        [[nodiscard]] bool has_loss() const noexcept { return loss_descriptor_.has_value(); }

        [[nodiscard]] torch::optim::Optimizer& optimizer() {
            if (!optimizer_) {
                throw std::logic_error("Optimizer has not been configured.");
            }
            return *optimizer_;
        }

        template <class Config, class Dataset = Core::SupervisedDataset>
        void train(Dataset dataset) {
            static_assert(Config::batch_size > 0, "Batch size must be greater than zero.");

            if (dataset.empty()) {
                return;
            }

            TrainOptions options{};
            options.epoch = Config::epochs;
            options.batch_size = Config::batch_size;
            options.shuffle = Config::shuffle;
            options.buffer_vram = Config::buffer_vram;
            options.monitor = false;

            auto packed = TrainingDetails::pack_dataset(std::move(dataset));
            train(std::move(packed.inputs), std::move(packed.targets), options);
        }

        void train(torch::Tensor train_inputs,
                   torch::Tensor train_targets,
                   TrainOptions options = {})
        {

            if (!has_optimizer()) {
                throw std::logic_error("Cannot train without an optimizer.");
            }
            if (!has_loss()) {
                throw std::logic_error("Cannot train without a loss function.");
            }
            if (!train_inputs.defined() || !train_targets.defined()) {
                throw std::invalid_argument("Training tensors must be defined.");
            }

            if (train_inputs.dim() == 0 || train_targets.dim() == 0) {
                throw std::invalid_argument("Training tensors must not be scalars.");
            }

            if (train_inputs.size(0) != train_targets.size(0)) {
                throw std::invalid_argument("Mismatched number of training samples between inputs and targets.");
            }

            if (options.batch_size == 0) {
                throw std::invalid_argument("Batch size must be greater than zero.");
            }

            clear_training_telemetry();

            if (options.epoch == 0) {
                return;
            }

            if (options.buffer_vram > 0 && !device_.is_cuda()) {
                throw std::runtime_error("VRAM buffering requires the model to be on a CUDA device.");
            }

            const auto total_samples = train_inputs.size(0);
            if (total_samples == 0) {
                return;
            }

            torch::nn::Module::train();
            this->to(device_);

            TrainOptions effective_options = options;
            if (effective_options.stream == nullptr) {
                effective_options.monitor = false;
            }
            auto training_dataset = TrainingDetails::prepare_tensor_dataset(std::move(train_inputs), std::move(train_targets));
            std::optional<typename TrainingDetails::TensorDataset> test_dataset{};
            auto build_evaluation_dataset = [&](const std::pair<torch::Tensor, torch::Tensor>& dataset,
                                                std::string_view name) {
                if (!dataset.first.defined() || !dataset.second.defined()) {
                    throw std::invalid_argument(std::string(name) + " tensors must be defined when provided.");
                }
                if (dataset.first.size(0) != dataset.second.size(0)) {
                    throw std::invalid_argument("Mismatched number of " + std::string(name)
                                                + " samples between inputs and targets.");
                }
                return TrainingDetails::prepare_tensor_dataset(dataset.first, dataset.second);
            };

            if (options.test.has_value()) {
                test_dataset = build_evaluation_dataset(*options.test, "test");
            } else if (options.validation.has_value()) {
                test_dataset = build_evaluation_dataset(*options.validation, "validation");
            }

            const bool use_buffer = effective_options.buffer_vram > 0;

            training_dataset = TrainingDetails::ensure_contiguous(std::move(training_dataset));
            training_dataset = TrainingDetails::ensure_cpu(std::move(training_dataset));

            if (test_dataset) {
                *test_dataset = TrainingDetails::ensure_contiguous(std::move(*test_dataset));
                *test_dataset = TrainingDetails::ensure_cpu(std::move(*test_dataset));
            }


            if (effective_options.shuffle) {
                if (use_buffer) {
                    TrainingDetails::run_epochs<true, true>(*this, training_dataset, test_dataset, effective_options);
                } else {
                    TrainingDetails::run_epochs<false, true>(*this, training_dataset, test_dataset, effective_options);

                }
            } else {
                if (use_buffer) {
                    TrainingDetails::run_epochs<true, false>(*this, training_dataset, test_dataset, effective_options);
                } else {
                    TrainingDetails::run_epochs<false, false>(*this, training_dataset, test_dataset, effective_options);
                }
            }

        }
    private:
        void reset_graph_shape_cache(GraphMode mode) const;
        void ensure_graph_input_shape(GraphMode mode, const torch::Tensor& tensor) const;
        void ensure_graph_batch_shapes(GraphMode mode,
                                       const torch::Tensor& inputs,
                                       const torch::Tensor& targets) const;
        void ensure_graph_replay_ready(GraphMode mode) const;
        void enforce_graph_shape(GraphMode mode,
                                 const torch::Tensor& tensor,
                                 std::optional<std::vector<int64_t>>& storage,
                                 std::string_view tensor_label) const;
        static std::vector<int64_t> tensor_shape_vector(const torch::Tensor& tensor);
        static std::string format_shape_vector(const std::vector<int64_t>& shape);
        static std::string describe_activation(Activation::Type type);
        static std::string describe_module(const Layer::Details::RegisteredLayer& layer);
        void reset_graph_shape_cache(GraphMode mode) const
        {
            if (mode == GraphMode::Capture) {
                last_input_shape_.reset();
                last_target_shape_.reset();
            }
        }

        void ensure_graph_input_shape(GraphMode mode, const torch::Tensor& tensor) const
        {
            enforce_graph_shape(mode, tensor, last_input_shape_, "input");
        }

        void ensure_graph_batch_shapes(GraphMode mode,
                                       const torch::Tensor& inputs,
                                       const torch::Tensor& targets) const
        {
            enforce_graph_shape(mode, inputs, last_input_shape_, "input");
            enforce_graph_shape(mode, targets, last_target_shape_, "target");
        }

        void ensure_graph_replay_ready(GraphMode mode) const
        {
            if (mode != GraphMode::Replay) {
                return;
            }
            if (!last_input_shape_ || !last_target_shape_) {
                throw std::logic_error(
                    "Graph replay requested before a capture pass recorded the training batch shapes.");
            }
        }

        void enforce_graph_shape(GraphMode mode,
                                 const torch::Tensor& tensor,
                                 std::optional<std::vector<int64_t>>& storage,
                                 std::string_view tensor_label) const
        {
            if (mode == GraphMode::Disabled) {
                return;
            }
            if (!tensor.defined()) {
                throw std::invalid_argument(
                    std::string("Graph optimisation requires a defined ") + std::string(tensor_label)
                    + " tensor during " + (mode == GraphMode::Capture ? "capture." : "replay."));
            }

            auto shape = tensor_shape_vector(tensor);

            if (mode == GraphMode::Capture) {
                if (!storage.has_value()) {
                    storage = shape;
                } else if (*storage != shape) {
                    throw std::invalid_argument(
                        "Graph capture observed inconsistent " + std::string(tensor_label)
                        + " tensor shapes. Expected " + format_shape_vector(*storage)
                        + " but received " + format_shape_vector(shape) + ".");
                }
                return;
            }

            if (!storage.has_value()) {
                throw std::logic_error(
                    "Graph replay requested for " + std::string(tensor_label)
                    + " tensors before a capture pass recorded the shape.");
            }

            if (*storage != shape) {
                throw std::invalid_argument(
                    "Graph replay expected " + std::string(tensor_label) + " tensor shape "
                    + format_shape_vector(*storage) + " but received " + format_shape_vector(shape) + ".");
            }
        }

        static std::vector<int64_t> tensor_shape_vector(const torch::Tensor& tensor)
        {
            std::vector<int64_t> shape;
            const auto sizes = tensor.sizes();
            shape.reserve(static_cast<std::size_t>(sizes.size()));
            for (int64_t index = 0; index < sizes.size(); ++index) {
                shape.push_back(sizes[index]);
            }
            return shape;
        }

        static std::string format_shape_vector(const std::vector<int64_t>& shape)
        {
            std::ostringstream stream;
            stream << '(';
            for (std::size_t index = 0; index < shape.size(); ++index) {
                if (index > 0) {
                    stream << ", ";
                }
                stream << shape[index];
            }
            stream << ')';
            return stream.str();
        }

        static void copy_tensor_into(torch::Tensor& destination, const torch::Tensor& source)
        {
            if (!source.defined()) {
                destination = torch::Tensor{};
                return;
            }

            if (!destination.defined()
                || destination.sizes() != source.sizes()
                || destination.scalar_type() != source.scalar_type()
                || destination.device() != source.device()) {
                destination = torch::empty_like(source);
                }

            destination.copy_(source);
        }

        void record_epoch_telemetry(TrainingTelemetry::EpochSnapshot snapshot)
        {
            telemetry_.append_epoch(std::move(snapshot));
        }

        void record_dataset_loss_telemetry(TrainingTelemetry::DatasetLossSnapshot snapshot)
        {
            telemetry_.append_dataset_loss(std::move(snapshot));
        }

        [[nodiscard]] std::vector<double> collect_learning_rates()
        {
            std::vector<double> learning_rates;

            auto append_from = [&](torch::optim::Optimizer* optimizer) {
                if (!optimizer) {
                    return;
                }
                for (auto& group : optimizer->param_groups()) {
                    learning_rates.push_back(group.options().get_lr());
                }
            };

            append_from(optimizer_.get());
            for (auto& optimizer : local_optimizers_) {
                append_from(optimizer.get());
            }

            return learning_rates;
        }



        void copy_into_graph_input_buffer(torch::Tensor tensor)
        {
            constexpr std::size_t kInputNodeIndex = 0;
            copy_tensor_into(graph_workspace_.input, tensor);
            graph_workspace_.bind_input(kInputNodeIndex);
        }

        [[nodiscard]] const torch::Tensor& graph_output_tensor() const noexcept
        {
            return graph_workspace_.output;
        }



        void clear_compiled_graph() noexcept
        {
            routing_active_ = false;
            compiled_nodes_.clear();
            compiled_steps_.clear();
            join_buffers_.clear();
            compiled_links_.clear();
            compiled_output_node_index_.reset();
            invalidate_execution_workspace();
        }


        void register_layer_runtime(const Layer::Details::RegisteredLayer& layer)
        {
            layer_parameters_.push_back(collect_layer_parameters(layer));
            layer_regularization_bindings_.push_back(
                bind_local_regularization(layer.local.regularization, layer_parameters_.back()));
            invalidate_execution_workspace();
        }

        void invalidate_execution_workspace() noexcept
        {
            execution_workspace_dirty_ = true;
            graph_workspace_.invalidate();
            cached_layer_pointers_.clear();
        }

        void ensure_execution_workspace()
        {
            if (execution_workspace_dirty_ || cached_layer_pointers_.size() != layers_.size()) {
                cached_layer_pointers_.resize(layers_.size());
                for (std::size_t index = 0; index < layers_.size(); ++index) {
                    cached_layer_pointers_[index] = &layers_[index];
                }
            }

            if (!has_compiled_routing() || compiled_nodes_.empty()) {
                execution_workspace_dirty_ = false;
                return;
            }

            auto& workspace = graph_workspace_;

            if (execution_workspace_dirty_ || workspace.node_buffers.size() != compiled_nodes_.size()) {
                workspace.ensure_node_capacity(compiled_nodes_.size());
            }

            workspace.ensure_join_scratch(join_buffers_);

            constexpr std::size_t kInputNodeIndex = 0;
            workspace.bind_input(kInputNodeIndex);

            const auto output_index = compiled_output_node_index_.value_or(
                compiled_steps_.empty() ? kInputNodeIndex : compiled_steps_.back().node_index);
            workspace.bind_output(output_index);

            for (auto& scratch : workspace.join_scratch) {
                scratch.clear();
            }

            execution_workspace_dirty_ = false;
        }

        static std::vector<torch::Tensor> collect_layer_parameters(const Layer::Details::RegisteredLayer& layer)
        {
            std::vector<torch::Tensor> parameters;
            if (!layer.module) {
                return parameters;
            }

            for (auto& parameter : layer.module->parameters()) {
                if (parameter.requires_grad()) {
                    parameters.push_back(parameter);
                }
            }

            return parameters;
        }

        std::vector<torch::Tensor> collect_global_trainable_parameters() const
        {
            std::vector<torch::Tensor> parameters;
            for (auto& parameter : this->parameters()) {
                if (parameter.requires_grad()) {
                    parameters.push_back(parameter);
                }
            }
            return parameters;
        }

        RegularizationStateStorage prepare_regularization_states(const Regularization::Descriptor& descriptor,
                                                                 const std::vector<torch::Tensor>& parameters) const
        {
            return std::visit(
                [&](const auto& concrete_descriptor) -> RegularizationStateStorage {
                    using DescriptorType = std::decay_t<decltype(concrete_descriptor)>;

                    if constexpr (std::is_same_v<DescriptorType, Regularization::EWCDescriptor>
                                  || std::is_same_v<DescriptorType, Regularization::MASDescriptor>
                                  || std::is_same_v<DescriptorType, Regularization::SIDescriptor>) {
                        auto storage = std::make_shared<std::vector<RegularizationState>>();
                        storage->reserve(parameters.size());
                        for (const auto& parameter : parameters) {
                            if constexpr (std::is_same_v<DescriptorType, Regularization::EWCDescriptor>) {
                                Regularization::Details::EWCState state{};
                                state.reference = parameter.detach().clone();
                                state.fisher_information = torch::zeros_like(parameter);
                                storage->emplace_back(std::move(state));
                            } else if constexpr (std::is_same_v<DescriptorType, Regularization::MASDescriptor>) {
                                Regularization::Details::MASState state{};
                                state.reference = parameter.detach().clone();
                                state.importance = torch::zeros_like(parameter);
                                storage->emplace_back(std::move(state));
                            } else if constexpr (std::is_same_v<DescriptorType, Regularization::SIDescriptor>) {
                                Regularization::Details::SIState state{};
                                state.reference = parameter.detach().clone();
                                state.importance = torch::zeros_like(parameter);
                                storage->emplace_back(std::move(state));
                            }
                        }
                        return storage;
                            } else if constexpr (std::is_same_v<DescriptorType, Regularization::SWAGDescriptor>) {
                                auto storage = std::make_shared<std::vector<RegularizationState>>();
                                storage->reserve(parameters.size());
                                for (std::size_t index = 0; index < parameters.size(); ++index) {
                                    storage->emplace_back(Regularization::Details::SWAGState{});
                                }
                                return storage;

                    } else {
                        return {};
                    }
                },
                descriptor);
        }

        RegularizationBinding make_regularization_binding(Regularization::Descriptor descriptor,
                                                          const std::vector<torch::Tensor>& parameters) const
        {
            RegularizationBinding binding{};
            binding.descriptor = std::move(descriptor);
            binding.states = prepare_regularization_states(binding.descriptor, parameters);
            binding.accumulator = Regularization::bind_accumulator(binding.descriptor, binding.states);
            return binding;
        }

        std::vector<RegularizationBinding> bind_local_regularization(
            const std::vector<Regularization::Descriptor>& descriptors,
            const std::vector<torch::Tensor>& parameters) const
        {
            std::vector<RegularizationBinding> bindings;
            bindings.reserve(descriptors.size());
            for (const auto& descriptor : descriptors) {
                bindings.push_back(make_regularization_binding(descriptor, parameters));
            }
            return bindings;
        }

        void update_regularization_binding_states(RegularizationBinding& binding,
                                                  const std::vector<torch::Tensor>& parameters,
                                                  std::size_t step_index)
        {
            if (parameters.empty()) {
                return;
            }

            std::visit(
                [&](auto& concrete_descriptor) {
                    using DescriptorType = std::decay_t<decltype(concrete_descriptor)>;
                    if constexpr (std::is_same_v<DescriptorType, Regularization::SWAGDescriptor>) {
                        const auto& options = concrete_descriptor.options;
                        if (options.coefficient == 0.0) {
                            return;
                        }

                        const std::size_t stride = options.accumulation_stride == 0 ? std::size_t{1}
                                                                                       : options.accumulation_stride;
                        if (step_index < options.start_step) {
                            return;
                        }
                        const auto adjusted_step = step_index - options.start_step;
                        if (adjusted_step % stride != 0) {
                            return;
                        }
                        if (!binding.states) {
                            return;
                        }

                        auto& state_storage = *binding.states;
                        const auto limit = std::min(parameters.size(), state_storage.size());
                        for (std::size_t index = 0; index < limit; ++index) {
                            auto& state_variant = state_storage[index];
                            if (!std::holds_alternative<Regularization::Details::SWAGState>(state_variant)) {
                                continue;
                            }

                            auto& state = std::get<Regularization::Details::SWAGState>(state_variant);
                            if (options.max_snapshots > 0 && state.snapshot_count >= options.max_snapshots) {
                                continue;
                            }

                            auto snapshot = parameters[index].detach();
                            if (!snapshot.defined()) {
                                continue;
                            }

                            auto snapshot_tensor = snapshot.clone();
                            if (!snapshot_tensor.defined()) {
                                continue;
                            }

                            if (state.snapshot_count == 0 || !state.mean.defined()) {
                                state.mean = snapshot_tensor;
                                state.variance = torch::zeros_like(state.mean);
                                state.snapshot_count = 1;
                                continue;
                            }

                            if (!state.variance.defined()) {
                                state.variance = torch::zeros_like(state.mean);
                            }

                            if (snapshot_tensor.device() != state.mean.device()) {
                                snapshot_tensor = snapshot_tensor.to(state.mean.device());
                            }
                            if (snapshot_tensor.scalar_type() != state.mean.scalar_type()) {
                                snapshot_tensor = snapshot_tensor.to(state.mean.scalar_type());
                            }

                            auto delta = snapshot_tensor - state.mean;
                            const double next_count = static_cast<double>(state.snapshot_count + 1);
                            state.mean = state.mean + delta / next_count;
                            auto delta2 = snapshot_tensor - state.mean;
                            state.variance = state.variance + delta * delta2;
                            state.snapshot_count += 1;
                        }
                    }
                },
                binding.descriptor);
        }

        void update_regularization_states(std::size_t step_index, bool regularization_active = false)
        {
            if (!regularization_active && !has_regularization()) {
                return;
            }

            for (auto& binding : global_regularization_bindings_) {
                update_regularization_binding_states(binding, global_regularization_parameters_, step_index);
            }

            for (std::size_t index = 0; index < layer_regularization_bindings_.size(); ++index) {
                auto& bindings = layer_regularization_bindings_[index];
                auto& parameters = layer_parameters_[index];
                for (auto& binding : bindings) {
                    update_regularization_binding_states(binding, parameters, step_index);
                }
            }
        }



        struct TrainingDetails {
            struct TensorDataset {
                torch::Tensor inputs;
                torch::Tensor targets;
            };

            [[nodiscard]] static TensorDataset prepare_tensor_dataset(torch::Tensor inputs,
                                                                      torch::Tensor targets)
            {
                return TensorDataset{std::move(inputs).contiguous(), std::move(targets).contiguous()};
            }

            [[nodiscard]] static TensorDataset ensure_contiguous(TensorDataset dataset)
            {
                dataset.inputs = dataset.inputs.contiguous();
                dataset.targets = dataset.targets.contiguous();
                return dataset;
            }

            [[nodiscard]] static TensorDataset ensure_cpu(TensorDataset dataset)
            {
                if (!dataset.inputs.device().is_cpu()) {
                    dataset.inputs = dataset.inputs.to(torch::kCPU);
                }
                if (!dataset.targets.device().is_cpu()) {
                    dataset.targets = dataset.targets.to(torch::kCPU);
                }
                return dataset;
            }

            template <class Dataset>
            [[nodiscard]] static TensorDataset pack_dataset(Dataset dataset)
            {
                if (dataset.empty()) {
                    return {};
                }

                std::vector<torch::Tensor> inputs;
                std::vector<torch::Tensor> targets;
                inputs.reserve(dataset.size());
                targets.reserve(dataset.size());

                for (auto& sample : dataset) {
                    inputs.push_back(std::move(sample.first));
                    targets.push_back(std::move(sample.second));
                }

                return TensorDataset{torch::stack(std::move(inputs)), torch::stack(std::move(targets))};
            }

            template <bool BufferVRAM, bool ShouldShuffle>
            static void run_epochs(Model& model,
                                   TensorDataset& train_dataset,
                                   const std::optional<TensorDataset>& test_dataset,
                                   const TrainOptions& options)
            {
                const auto device = model.device();
                const auto total_samples = train_dataset.inputs.size(0);
                const auto batch_size = static_cast<std::int64_t>(options.batch_size);
                const auto graph_mode = options.graph_mode;
                const bool graph_mode_active = graph_mode != GraphMode::Disabled;

                if (graph_mode == GraphMode::Capture) {
                    model.reset_graph_shape_cache(graph_mode);
                } else if (graph_mode == GraphMode::Replay) {
                    model.ensure_graph_replay_ready(graph_mode);
                }

                torch::TensorOptions index_options = torch::TensorOptions().dtype(torch::kLong);


                auto best_test = std::optional<double>{};
                std::vector<torch::Tensor> best_parameters;
                std::vector<torch::Tensor> best_buffers;
                bool best_state_captured = false;

                std::size_t step_index = 0;

                const bool regularization_active = model.has_regularization();

                for (std::size_t epoch = 0; epoch < options.epoch; ++epoch) {
                    const auto epoch_start = std::chrono::steady_clock::now();

                    torch::Tensor epoch_indices;
                    if constexpr (ShouldShuffle) {
                        if (total_samples > 1) {
                            epoch_indices = torch::randperm(total_samples, index_options);
                        } else {
                            epoch_indices = torch::arange(total_samples, index_options);
                        }
                    }

                    auto accumulation = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
                    std::int64_t weight = 0;

                    const std::size_t total_batches = total_samples > 0
                        ? static_cast<std::size_t>((total_samples + batch_size - 1) / batch_size)
                        : 0;

                    auto fetch_batch = [&](std::size_t batch_index) {
                        const auto offset = static_cast<std::int64_t>(batch_index) * batch_size;
                        const auto remaining = total_samples - offset;
                        const auto current_batch = std::min<std::int64_t>(batch_size, remaining);
                        if (current_batch <= 0) {
                            return std::pair<torch::Tensor, torch::Tensor>{torch::Tensor{}, torch::Tensor{}};
                        }
                        if (graph_mode_active && current_batch != batch_size) {
                            throw std::invalid_argument(
                                "Graph optimisation requires every batch to match the captured batch size ("
                                + std::to_string(batch_size)
                                + "). Received " + std::to_string(current_batch)
                                + " samples; pad or drop the remainder before enabling graph replay.");
                        }


                        torch::Tensor batch_inputs;
                        torch::Tensor batch_targets;

                        if constexpr (ShouldShuffle) {
                            auto batch_indices = epoch_indices.narrow(0, offset, current_batch);
                            if (!batch_indices.device().is_cpu()) {
                                batch_indices = batch_indices.to(torch::kCPU);
                            }
                            batch_inputs = train_dataset.inputs.index_select(0, batch_indices);
                            batch_targets = train_dataset.targets.index_select(0, batch_indices);
                        } else {
                            batch_inputs = train_dataset.inputs.narrow(0, offset, current_batch);
                            batch_targets = train_dataset.targets.narrow(0, offset, current_batch);
                        }

                        if (batch_inputs.defined() && batch_inputs.device() != device) {
                            batch_inputs = batch_inputs.to(device);
                        }
                        if (batch_targets.defined() && batch_targets.device() != device) {
                            batch_targets = batch_targets.to(device);
                        }

                        return std::pair<torch::Tensor, torch::Tensor>{std::move(batch_inputs), std::move(batch_targets)};
                    };

                    auto process_batch = [&](torch::Tensor batch_inputs, torch::Tensor batch_targets) {
                        if (!batch_inputs.defined() || !batch_targets.defined()) {
                            return std::int64_t{0};
                        }

                        const auto current_batch = batch_targets.size(0);
                        if (current_batch <= 0) {
                            return std::int64_t{0};
                        }


                        if (graph_mode_active) {
                            model.ensure_graph_batch_shapes(graph_mode, batch_inputs, batch_targets);
                        }

                        model.zero_grad();
                        ForwardOptions forward_options{};
                        forward_options.graph_mode = graph_mode;
                        auto prediction = model.forward(std::move(batch_inputs), std::move(forward_options));

                        if (!prediction.sizes().equals(batch_targets.sizes())) {
                            if (batch_targets.numel() == prediction.numel()) {
                                batch_targets = batch_targets.reshape_as(prediction);
                            }
                        }

                        auto loss = model.compute_loss(prediction, batch_targets);
                        if (loss.dim() != 0) {
                            loss = loss.mean();
                        }
                        if (regularization_active) {
                            auto regularization_penalty = model.compute_regularization_penalty();
                            if (regularization_penalty.defined()) {
                                if (regularization_penalty.device() != loss.device()) {
                                    regularization_penalty = regularization_penalty.to(loss.device());
                                }
                                if (regularization_penalty.scalar_type() != loss.scalar_type()) {
                                    regularization_penalty = regularization_penalty.to(loss.scalar_type());
                                }
                                loss = loss + regularization_penalty;
                            }
                        }

                        loss.backward();
                        model.step();


                        if (regularization_active) {
                            model.update_regularization_states(step_index, true);
                        }
                        ++step_index;


                        auto loss_tensor = loss.detach();
                        if (loss_tensor.device() != accumulation.device()) {
                            loss_tensor = loss_tensor.to(accumulation.device());
                        }
                        loss_tensor = loss_tensor.to(torch::kFloat64);
                        loss_tensor.mul_(static_cast<double>(current_batch));
                        accumulation.add_(loss_tensor);
                        return current_batch;
                    };

                    if constexpr (BufferVRAM) {
                        if (graph_mode_active) {
                            for (std::size_t batch_index = 0; batch_index < total_batches; ++batch_index) {
                                auto batch_pair = fetch_batch(batch_index);
                                const auto processed = process_batch(std::move(batch_pair.first), std::move(batch_pair.second));
                                weight += processed;
                            }
                        } else {
                            std::deque<std::pair<torch::Tensor, torch::Tensor>> buffered_batches;
                            std::size_t next_batch_to_load = 0;
                            const std::size_t max_batches = total_batches == 0 ? 1 : total_batches;
                            const std::size_t buffer_limit = std::max<std::size_t>(1,
                                std::min<std::size_t>(options.buffer_vram + 1, max_batches));

                            auto maintain_buffer = [&](std::size_t current_index) {
                                const std::size_t desired_size = std::min(buffer_limit, total_batches - current_index);
                                while (buffered_batches.size() < desired_size && next_batch_to_load < total_batches) {
                                    auto batch = fetch_batch(next_batch_to_load);
                                    buffered_batches.push_back(std::move(batch));
                                    ++next_batch_to_load;
                                }
                            };

                            for (std::size_t batch_index = 0; batch_index < total_batches; ++batch_index) {
                                maintain_buffer(batch_index);
                                if (buffered_batches.empty()) {
                                    break;
                                }


                                auto batch_pair = std::move(buffered_batches.front());
                                buffered_batches.pop_front();

                                const auto processed = process_batch(std::move(batch_pair.first), std::move(batch_pair.second));
                                weight += processed;


                                maintain_buffer(batch_index + 1);
                            }
                        }
                    } else {
                        for (std::size_t batch_index = 0; batch_index < total_batches; ++batch_index) {
                            auto batch_pair = fetch_batch(batch_index);
                            const auto processed = process_batch(std::move(batch_pair.first), std::move(batch_pair.second));
                            weight += processed;
                        }
                    }

                    const auto train_loss = weight > 0
                        ? accumulation.item<double>() / static_cast<double>(weight)
                        : 0.0;

                    std::optional<double> test_loss{};
                    if (test_dataset) {
                        test_loss = compute_dataset_loss<BufferVRAM>(model, *test_dataset, options.batch_size);
                    }

                    bool improved = false;
                    std::optional<double> delta{};
                    if (test_loss) {
                        if (!best_test) {
                            improved = true;
                            best_test = test_loss;
                        } else {
                            const auto previous_best = *best_test;
                            delta = *test_loss - previous_best;
                            if (*test_loss < previous_best) {
                                improved = true;
                                best_test = test_loss;
                            }
                        }
                    }

                    if (improved && options.restore_best_state) {
                        best_parameters.clear();
                        best_buffers.clear();
                        best_parameters.reserve(model.parameters().size());
                        best_buffers.reserve(model.buffers().size());

                        for (auto& parameter : model.parameters()) {
                            if (parameter.defined()) {
                                best_parameters.push_back(parameter.detach().clone());
                            } else {
                                best_parameters.push_back({});
                            }
                        }

                        for (auto& buffer : model.buffers()) {
                            if (buffer.defined()) {
                                best_buffers.push_back(buffer.detach().clone());
                            } else {
                                best_buffers.push_back({});
                            }
                        }

                        best_state_captured = true;
                    }


                    const auto duration_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_start).count();

                    const auto epoch_timestamp = std::chrono::system_clock::now();
                    auto learning_rates = model.collect_learning_rates();
                    model.record_epoch_telemetry({
                        epoch + 1,
                        train_loss,
                        test_loss,
                        delta,
                        std::move(learning_rates),
                        epoch_timestamp,
                        duration_seconds
                    });

                    if (options.monitor && options.stream) {
                        log_epoch(*options.stream,
                                  epoch + 1,
                                  options.epoch,
                                  train_loss,
                                  test_loss,
                                  delta,
                                  improved,
                                  duration_seconds);
                    }
                }
                if (options.restore_best_state && best_state_captured) {
                    std::cout << "[Thot] Reloading best state of the network..." << std::endl;
                    auto parameters = model.parameters();
                    const auto parameter_limit = std::min(parameters.size(), best_parameters.size());
                    for (std::size_t index = 0; index < parameter_limit; ++index) {
                        auto& target = parameters[index];
                        const auto& source = best_parameters[index];
                        if (target.defined() && source.defined()) {
                            target.detach_();
                            target.copy_(source);
                        }
                    }

                    auto buffers = model.buffers();
                    const auto buffer_limit = std::min(buffers.size(), best_buffers.size());
                    for (std::size_t index = 0; index < buffer_limit; ++index) {
                        auto& target = buffers[index];
                        const auto& source = best_buffers[index];
                        if (target.defined() && source.defined()) {
                            target.detach_();
                            target.copy_(source);
                        }
                    }
                }
            }


            template <bool BufferVRAM>
            static std::optional<double> compute_dataset_loss(Model& model, const TensorDataset& dataset, std::size_t batch_size) {
                if (!dataset.inputs.defined() || !dataset.targets.defined()) {
                    return std::nullopt;
                }
                if (dataset.inputs.size(0) == 0) {
                    return std::nullopt;
                }

                const auto device = model.device();
                const auto total_samples = dataset.inputs.size(0);
                const auto local_batch = static_cast<std::int64_t>(batch_size);
                const bool regularization_active = model.has_regularization();

                torch::NoGradGuard no_grad;
                const bool was_training = model.is_training();
                model.eval();

                auto accumulation = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
                std::int64_t weight = 0;

                for (std::int64_t offset = 0; offset < total_samples; offset += local_batch) {
                    const auto remaining = total_samples - offset;
                    const auto current_batch = std::min<std::int64_t>(local_batch, remaining);
                    if (current_batch <= 0) {
                        break;
                    }
                    torch::Tensor inputs;
                    torch::Tensor targets;

                    inputs = dataset.inputs.narrow(0, offset, current_batch);
                    targets = dataset.targets.narrow(0, offset, current_batch);

                    if (inputs.device() != device) {
                        inputs = inputs.to(device);
                    }
                    if (targets.device() != device) {
                        targets = targets.to(device);
                    }

                    auto prediction = model.forward(inputs);

                    if (!prediction.sizes().equals(targets.sizes())) {
                        if (targets.numel() == prediction.numel()) {
                            targets = targets.reshape_as(prediction);
                        }
                    }

                    auto loss = model.compute_loss(prediction, targets);
                    if (loss.dim() != 0) {
                        loss = loss.mean();
                    }
                    if (regularization_active) {
                        auto regularization_penalty = model.compute_regularization_penalty();
                        if (regularization_penalty.defined()) {
                            if (regularization_penalty.device() != loss.device()) {
                                regularization_penalty = regularization_penalty.to(loss.device());
                            }
                            if (regularization_penalty.scalar_type() != loss.scalar_type()) {
                                regularization_penalty = regularization_penalty.to(loss.scalar_type());
                            }
                            loss = loss + regularization_penalty;
                        }
                    }

                    auto loss_tensor = loss.detach();
                    if (loss_tensor.device() != accumulation.device()) {
                        loss_tensor = loss_tensor.to(accumulation.device());
                    }
                    loss_tensor = loss_tensor.to(torch::kFloat64);
                    loss_tensor.mul_(static_cast<double>(current_batch));
                    accumulation.add_(loss_tensor);
                    weight += current_batch;
                }
                if (was_training) {
                    model.train();
                } else {
                    model.eval();
                }

                if (weight == 0) {
                    return std::nullopt;
                }

                const double averaged_loss = accumulation.item<double>() / static_cast<double>(weight);
                auto learning_rates = model.collect_learning_rates();
                const auto timestamp = std::chrono::system_clock::now();
                model.record_dataset_loss_telemetry({
                    averaged_loss,
                    static_cast<std::size_t>(dataset.inputs.size(0)),
                    std::move(learning_rates),
                    timestamp
                });

                return averaged_loss;
            }

            static void log_epoch(std::ostream& stream,
                                  std::size_t epoch_index,
                                  std::size_t total_epochs,
                                  double train_loss,
                                  const std::optional<double>& test_loss,
                                  const std::optional<double>& delta,
                                  bool improved,
                                  double duration_seconds)
            {
                using Utils::Terminal::ApplyColor;
                using Utils::Terminal::Colors::kBrightBlack;
                using Utils::Terminal::Colors::kBrightBlue;
                using Utils::Terminal::Colors::kBrightGreen;
                using Utils::Terminal::Colors::kBrightYellow;
                using Utils::Terminal::Colors::kReset;

                std::ostringstream line;
                line << "Epoch [" << epoch_index << "/" << total_epochs << "] | ";
                line << ApplyColor("Train", kBrightYellow) << " loss: "
                     << std::fixed << std::setprecision(6) << train_loss << " | ";
                line << ApplyColor("Test", kBrightBlue) << " loss: ";
                if (test_loss) {
                    line << std::fixed << std::setprecision(6) << *test_loss;
                } else {
                    line << "N/A";
                }

                line << " | ΔLoss: ";
                if (test_loss && delta) {
                    std::ostringstream delta_stream;
                    delta_stream << std::showpos << std::fixed << std::setprecision(6) << *delta;
                    line << delta_stream.str();
                } else if (test_loss) {
                    line << "N/A";
                } else {
                    line << "N/A";
                }

                const std::string nabla_symbol{"∇"};
                const std::string grey{kBrightBlack};
                const std::string green{kBrightGreen};
                const std::string reset{kReset};

                std::string nabla_indicator;

                if (improved)
                    line << grey << " (" << green << nabla_symbol <<grey << ")" << reset;
                else
                    line << grey << " (" << nabla_symbol << ")" << reset;

                std::ostringstream duration_stream;
                duration_stream << std::fixed << std::setprecision(2) << duration_seconds << "sec";
                line << " | "
                     << ApplyColor("duration: " + duration_stream.str(), kBrightBlack);

                stream << line.str() << '\n';
            }
        };

        std::vector<Layer::Details::RegisteredLayer> layers_{};
        std::vector<NamedModuleDescriptor> module_descriptors_{};
        std::vector<CalibrationMethod> calibration_methods_{};
        std::vector<std::vector<torch::Tensor>> layer_parameters_{};
        std::vector<std::vector<RegularizationBinding>> layer_regularization_bindings_{};
        std::vector<torch::Tensor> global_regularization_parameters_{};
        std::vector<RegularizationBinding> global_regularization_bindings_{};
        mutable std::optional<std::vector<int64_t>> last_input_shape_{};
        mutable std::optional<std::vector<int64_t>> last_target_shape_{};
        TrainingTelemetry telemetry_{};
        std::size_t module_index_{0};
        std::unordered_map<std::string, ModuleNameBinding> module_name_index_{};
        std::vector<CompiledNode> compiled_nodes_{};
        std::vector<CompiledStep> compiled_steps_{};
        std::vector<JoinBuffer> join_buffers_{};
        std::vector<LinkSpec> compiled_links_{};
        std::optional<std::size_t> compiled_output_node_index_{};
        std::vector<torch::Tensor> node_activations_{};
        std::vector<std::vector<torch::Tensor>> join_workspace_{};
        GraphExecutionWorkspace graph_workspace_{};
        std::vector<Layer::Details::RegisteredLayer*> cached_layer_pointers_{};
        bool execution_workspace_dirty_{true};
        bool routing_active_{false};
        std::unique_ptr<torch::optim::Optimizer> optimizer_{};
        std::vector<std::unique_ptr<torch::optim::Optimizer>> local_optimizers_{};
        std::unique_ptr<LrScheduler::Details::Scheduler> scheduler_{};
        using StepImpl = void (Model::*)();
        StepImpl step_impl_{&Model::step_not_configured};
        using LossDescriptor = std::variant<Loss::MSEDescriptor, Loss::CrossEntropyDescriptor>;
        std::optional<LossDescriptor> loss_descriptor_{};
        std::string name_{};
        torch::Device device_{torch::kCPU, 0};
        bool regularization_configured_{false};
        std::string model_name_{};



        void configure_step_impl() noexcept {
            if (!optimizer_ && local_optimizers_.empty()) {
                step_impl_ = &Model::step_not_configured;
                return;
            }
            step_impl_ = scheduler_ ? &Model::step_configured<true> : &Model::step_configured<false>;
        }

        void step_not_configured() {
            throw std::logic_error("Optimizer has not been configured.");
        }

        template <bool WithScheduler>
        void step_configured() {
            if constexpr (WithScheduler) {
                if (scheduler_) {
                    scheduler_->step();
                }
            }
            if (optimizer_) {
                optimizer_->step();
            }
            for (auto& optimizer : local_optimizers_) {
                optimizer->step();
            }
        }
        [[nodiscard]] std::size_t next_module_index() noexcept { return module_index_++; }

        void reset_runtime_state() {
            auto preserved_name = name_;
            auto preserved_device = device_;
            auto preserved_model_name = model_name_;

            this->~Model();
            new (this) Model(preserved_name);

            device_ = preserved_device;
            model_name_ = std::move(preserved_model_name);
            module_name_index_.clear();
            clear_compiled_graph();
        }

        static std::string format_tensor_shape(const torch::Tensor& tensor) {
            std::ostringstream stream;
            stream << '(';
            const auto sizes = tensor.sizes();
            for (int64_t index = 0; index < sizes.size(); ++index) {
                if (index > 0) {
                    stream << ", ";
                }
                stream << sizes[index];
            }
            stream << ')';
            return stream.str();
        }
    };
}

#include "plot/plot.hpp"

namespace Thot {
    template <class Descriptor, class... Args>
    decltype(auto) Model::plot(Descriptor descriptor, Args&&... args) {
        return Plot::Render(*this,
                            std::move(descriptor),
                            std::forward<Args>(args)...);
    }
}
#endif //THOT_CORE_HPP