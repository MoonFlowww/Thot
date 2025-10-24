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
#include <cstddef>

#include <cmath>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <torch/torch.h>


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



namespace Thot {
    namespace Core {
        template <bool BufferVRAM>
        struct DevicePolicy {
            [[nodiscard]] static torch::Device select() {
                if constexpr (BufferVRAM) {
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
                  bool BufferVRAM,
                  class DevicePolicyT = DevicePolicy<BufferVRAM>>
        struct TrainingConfig {
            static_assert(Epochs > 0, "TrainingConfig requires at least one epoch.");
            static_assert(BatchSize > 0, "TrainingConfig requires a positive batch size.");

            static constexpr std::size_t epochs = Epochs;
            static constexpr std::size_t batch_size = BatchSize;
            static constexpr bool shuffle = Shuffle;
            static constexpr bool buffer_vram = BufferVRAM;

            using DevicePolicy = DevicePolicyT;
        };

        using SupervisedSample = std::pair<torch::Tensor, torch::Tensor>;
        using SupervisedDataset = std::vector<SupervisedSample>;

        using DefaultTrainingConfig = TrainingConfig<10, 32, true, false>;
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

    struct TrainOptions {
        std::size_t epoch{Core::kDefaultTrainingConfig.epochs};
        std::size_t batch_size{Core::kDefaultTrainingConfig.batch_size};
        bool shuffle{Core::kDefaultTrainingConfig.shuffle};
        bool buffer_vram{Core::kDefaultTrainingConfig.buffer_vram};
        bool monitor{true};
        std::optional<std::pair<torch::Tensor, torch::Tensor>> validation{};
        std::optional<std::pair<torch::Tensor, torch::Tensor>> test{};
        std::ostream* stream{&std::cout};
    };

    class Model : public torch::nn::Module {

        using RegularizationState = Regularization::StateVariant;
        using RegularizationStateStorage = std::shared_ptr<std::vector<RegularizationState>>;
        using RegularizationAccumulator = Regularization::Accumulator;

        struct RegularizationBinding {
            Regularization::Descriptor descriptor{};
            RegularizationStateStorage states{};
            RegularizationAccumulator accumulator{};
        };


    public:
        Model() = default;

        using torch::nn::Module::train;

        using ModuleDescriptor = std::variant<Layer::Descriptor, Block::Descriptor>;
        void add(ModuleDescriptor descriptor) {
            if (regularization_configured_)
                throw std::logic_error("Cannot add modules after regularization has been configured.");
            auto register_layer = [this](auto&& concrete_descriptor) {
                using DescriptorType = std::decay_t<decltype(concrete_descriptor)>;
                auto registered = Layer::Details::build_registered_layer(
                    *this,
                    static_cast<const DescriptorType&>(concrete_descriptor),
                    next_module_index());
                layers_.push_back(std::move(registered));
                register_layer_runtime(layers_.back());
            };

            auto handle_layer_descriptor = [&](Layer::Descriptor layer_descriptor) {
                switch (layer_descriptor.index()) {
                    case 0:
                        register_layer(std::get<Layer::FCDescriptor>(std::move(layer_descriptor)));
                        break;
                    case 1:
                        register_layer(std::get<Layer::Conv2dDescriptor>(std::move(layer_descriptor)));
                        break;
                    case 2:
                        register_layer(std::get<Layer::BatchNorm2dDescriptor>(std::move(layer_descriptor)));
                        break;
                    case 3:
                        register_layer(std::get<Layer::PoolingDescriptor>(std::move(layer_descriptor)));
                        break;
                    case 4:
                        register_layer(std::get<Layer::DropoutDescriptor>(std::move(layer_descriptor)));
                        break;
                    case 5:
                        register_layer(std::get<Layer::FlattenDescriptor>(std::move(layer_descriptor)));
                        break;
                    default:
                        throw std::invalid_argument("Unsupported layer descriptor passed to Model::add.");
                }
            };

            switch (descriptor.index()) {
                case 0:
                    handle_layer_descriptor(std::get<Layer::Descriptor>(std::move(descriptor)));
                    break;
                case 1: {
                    auto block_descriptor = std::get<Block::Descriptor>(std::move(descriptor));
                    switch (block_descriptor.index()) {
                        case 0: {
                            auto sequential = std::get<Block::SequentialDescriptor>(std::move(block_descriptor));
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

                                layers_.push_back(std::move(registered_layer));
                                register_layer_runtime(layers_.back());
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
                                    handle_layer_descriptor(std::move(layer));
                                }

                            }
                            break;
                        }
                        case 1: {
                            auto residual = std::get<Block::ResidualDescriptor>(std::move(block_descriptor));
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

                            layers_.push_back(std::move(registered_layer));
                            register_layer_runtime(layers_.back());
                            break;
                        }
                        case 2: {
                            auto encoder_descriptor = std::get<Block::Transformer::Classic::EncoderDescriptor>(std::move(block_descriptor));
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

                            layers_.push_back(std::move(registered_layer));
                            register_layer_runtime(layers_.back());
                            break;
                        }
                        case 3:
                            throw std::invalid_argument("Transformer decoder blocks are not yet supported by Model::add.");
                        default:
                            throw std::invalid_argument("Unsupported block descriptor passed to Model::add.");
                    }
                    break;
                }
                default:
                    throw std::invalid_argument("Unsupported module descriptor passed to Model::add.");
            }
        }

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
                total = total + penalty;
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


        [[nodiscard]] torch::Tensor forward(torch::Tensor input) {
            auto output = std::move(input);
            if (output.device() != device_)
                output = output.to(device_);
            for (auto& layer : layers_) {
                output = layer.forward(std::move(output));
                output = Activation::Details::apply(layer.activation, std::move(output));
            }
            return output;
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

        [[nodiscard]] auto evaluate(torch::Tensor evaluation_inputs, torch::Tensor evaluation_targets, Evaluation::ClassificationDescriptor descriptor, std::vector<Metric::Classification::Descriptor> metrics, Evaluation::Options options = {}) -> Evaluation::ClassificationReport {
            return Evaluation::Evaluate(
                *this,
                std::move(evaluation_inputs),
                std::move(evaluation_targets),
                descriptor,
                std::move(metrics),
                options);
        }

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

            if (options.epoch == 0) {
                return;
            }

            if (options.buffer_vram && !device_.is_cuda()) {
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

            if (effective_options.buffer_vram) {
                training_dataset = TrainingDetails::buffer_dataset(training_dataset, device_);
                if (test_dataset) {
                    *test_dataset = TrainingDetails::buffer_dataset(*test_dataset, device_);
                }
            } else {
                training_dataset = TrainingDetails::ensure_contiguous(std::move(training_dataset));
                if (test_dataset) {
                    *test_dataset = TrainingDetails::ensure_contiguous(std::move(*test_dataset));
                }
                if (training_dataset.inputs.device().is_cuda()) {
                    training_dataset.inputs = training_dataset.inputs.to(torch::kCPU);
                }
                if (training_dataset.targets.device().is_cuda()) {
                    training_dataset.targets = training_dataset.targets.to(torch::kCPU);
                }
                if (test_dataset) {
                    if (test_dataset->inputs.device().is_cuda()) {
                        test_dataset->inputs = test_dataset->inputs.to(torch::kCPU);
                    }
                    if (test_dataset->targets.device().is_cuda()) {
                        test_dataset->targets = test_dataset->targets.to(torch::kCPU);
                    }
                }
            }
            if (effective_options.buffer_vram && !training_dataset.inputs.device().is_cuda()) {
                training_dataset.inputs = training_dataset.inputs.to(device_);
                training_dataset.targets = training_dataset.targets.to(device_);
            }

            if (effective_options.buffer_vram && test_dataset && !test_dataset->inputs.device().is_cuda()) {
                test_dataset->inputs = test_dataset->inputs.to(device_);
                test_dataset->targets = test_dataset->targets.to(device_);
            }

            if (effective_options.shuffle) {
                if (effective_options.buffer_vram) {
                    TrainingDetails::run_epochs<true, true>(*this, training_dataset, test_dataset, effective_options);
                } else {
                    TrainingDetails::run_epochs<false, true>(*this, training_dataset, test_dataset, effective_options);

                }
            } else {
                if (effective_options.buffer_vram) {
                    TrainingDetails::run_epochs<true, false>(*this, training_dataset, test_dataset, effective_options);
                } else {
                    TrainingDetails::run_epochs<false, false>(*this, training_dataset, test_dataset, effective_options);
                }
            }
        }
    private:

        void register_layer_runtime(const Layer::Details::RegisteredLayer& layer)
        {
            layer_parameters_.push_back(collect_layer_parameters(layer));
            layer_regularization_bindings_.push_back(
                bind_local_regularization(layer.local.regularization, layer_parameters_.back()));
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

        void update_regularization_states(std::size_t step_index)
        {
            if (!has_regularization()) {
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

            [[nodiscard]] static TensorDataset buffer_dataset(TensorDataset dataset, const torch::Device& device)
            {
                if (dataset.inputs.device() != device) {
                    dataset.inputs = dataset.inputs.to(device);
                }
                if (dataset.targets.device() != device) {
                    dataset.targets = dataset.targets.to(device);
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

                torch::TensorOptions index_options = torch::TensorOptions().dtype(torch::kLong);
                if constexpr (BufferVRAM) {
                    index_options = index_options.device(device);
                }

                auto best_test = std::optional<double>{};

                std::size_t step_index = 0;

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

                    for (std::int64_t offset = 0; offset < total_samples; offset += batch_size) {
                        const auto remaining = total_samples - offset;
                        const auto current_batch = std::min<std::int64_t>(batch_size, remaining);
                        if (current_batch <= 0) {
                            break;
                        }

                        torch::Tensor batch_inputs;
                        torch::Tensor batch_targets;

                        if constexpr (ShouldShuffle) {
                            auto batch_indices = epoch_indices.narrow(0, offset, current_batch);
                            if constexpr (BufferVRAM) {
                                batch_inputs = train_dataset.inputs.index_select(0, batch_indices);
                                batch_targets = train_dataset.targets.index_select(0, batch_indices);
                            } else {
                                auto cpu_indices = batch_indices.to(torch::kCPU);
                                batch_inputs = train_dataset.inputs.index_select(0, cpu_indices);
                                batch_targets = train_dataset.targets.index_select(0, cpu_indices);
                            }
                        } else {
                            batch_inputs = train_dataset.inputs.narrow(0, offset, current_batch);
                            batch_targets = train_dataset.targets.narrow(0, offset, current_batch);
                        }

                        if constexpr (!BufferVRAM) {
                            if (batch_inputs.device() != device) {
                                batch_inputs = batch_inputs.to(device);
                            }
                            if (batch_targets.device() != device) {
                                batch_targets = batch_targets.to(device);
                            }
                        }

                        if constexpr (BufferVRAM) {
                            if (batch_inputs.device() != device) {
                                batch_inputs = batch_inputs.to(device);
                            }
                            if (batch_targets.device() != device) {
                                batch_targets = batch_targets.to(device);
                            }
                        }

                        model.zero_grad();
                        auto prediction = model.forward(batch_inputs);

                        if (!prediction.sizes().equals(batch_targets.sizes())) {
                            if (batch_targets.numel() == prediction.numel()) {
                                batch_targets = batch_targets.reshape_as(prediction);
                            }
                        }

                        auto loss = model.compute_loss(prediction, batch_targets);
                        if (loss.dim() != 0) {
                            loss = loss.mean();
                        }
                        if (model.has_regularization()) {
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


                        if (model.has_regularization()) {
                            model.update_regularization_states(step_index);
                        }
                        ++step_index;


                        auto loss_tensor = loss.detach();
                        if (loss_tensor.device() != accumulation.device()) {
                            loss_tensor = loss_tensor.to(accumulation.device());
                        }
                        loss_tensor = loss_tensor.to(torch::kFloat64);
                        loss_tensor.mul_(static_cast<double>(current_batch));
                        accumulation.add_(loss_tensor);
                        weight += current_batch;
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

                    const auto duration_seconds = std::chrono::duration<double>(
                                            std::chrono::steady_clock::now() - epoch_start).count();

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
            }


            template <bool BufferVRAM>
                        static std::optional<double> compute_dataset_loss(Model& model,
                                                                          const TensorDataset& dataset,
                                                                          std::size_t batch_size)
            {
                if (!dataset.inputs.defined() || !dataset.targets.defined()) {
                    return std::nullopt;
                }
                if (dataset.inputs.size(0) == 0) {
                    return std::nullopt;
                }

                const auto device = model.device();
                const auto total_samples = dataset.inputs.size(0);
                const auto local_batch = static_cast<std::int64_t>(batch_size);

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

                    if constexpr (!BufferVRAM) {
                        if (inputs.device() != device) {
                            inputs = inputs.to(device);
                        }
                        if (targets.device() != device) {
                            targets = targets.to(device);
                        }
                    } else {
                        if (inputs.device() != device) {
                            inputs = inputs.to(device);
                        }
                        if (targets.device() != device) {
                            targets = targets.to(device);
                        }
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
                    if (model.has_regularization()) {
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

                return accumulation.item<double>() / static_cast<double>(weight);
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
        std::vector<std::vector<torch::Tensor>> layer_parameters_{};
        std::vector<std::vector<RegularizationBinding>> layer_regularization_bindings_{};
        std::vector<torch::Tensor> global_regularization_parameters_{};
        std::vector<RegularizationBinding> global_regularization_bindings_{};
        std::size_t module_index_{0};
        std::unique_ptr<torch::optim::Optimizer> optimizer_{};
        std::vector<std::unique_ptr<torch::optim::Optimizer>> local_optimizers_{};
        std::unique_ptr<LrScheduler::Details::Scheduler> scheduler_{};
        using StepImpl = void (Model::*)();
        StepImpl step_impl_{&Model::step_not_configured};
        using LossDescriptor = std::variant<Loss::MSEDescriptor, Loss::CrossEntropyDescriptor>;
        std::optional<LossDescriptor> loss_descriptor_{};
        torch::Device device_{torch::kCPU, 0};
        bool regularization_configured_{false};



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
    };
}
#endif //THOT_CORE_HPP