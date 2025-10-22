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
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <torch/torch.h>

#include "activation/activation.hpp"
#include "activation/apply.hpp"
#include "initialization/initialization.hpp"
#include "initialization/apply.hpp"
#include "layer/layer.hpp"
#include "loss/loss.hpp"
#include "loss/details/mse.hpp"
#include "optimizer/optimizer.hpp"
#include "lrscheduler/lrscheduler.hpp"



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

    class Model : public torch::nn::Module {
    public:
        Model() = default;

        using torch::nn::Module::train;

        void add(Layer::Descriptor descriptor) {
            const auto index = layers_.size();
            auto registered_layer = std::visit(
                [&](auto&& concrete_descriptor) {
                    return Layer::Details::build_registered_layer(*this, concrete_descriptor, index);
                },
                std::move(descriptor));
            layers_.push_back(std::move(registered_layer));
        }

        void set_optimizer(Optimizer::Descriptor descriptor, std::optional<LrScheduler::Descriptor> scheduler = std::nullopt) {
            if (layers_.empty()) {
                throw std::logic_error("Cannot create optimizer before any layer has been registered.");
            }
            optimizer_ = std::visit(
                [&](const auto& concrete_descriptor) -> std::unique_ptr<torch::optim::Optimizer> {
                    return Optimizer::Details::build_optimizer(*this, concrete_descriptor);
                },
                std::move(descriptor));
            step_impl_ = &Model::step_without_scheduler;
            scheduler_.reset();
            if (scheduler.has_value()) {
                scheduler_ = std::visit(
                    [&](const auto& concrete_descriptor) -> std::unique_ptr<LrScheduler::Details::Scheduler> {
                        return LrScheduler::Details::build_scheduler(*this, *optimizer_, concrete_descriptor);
                    }, std::move(*scheduler));
                if (scheduler_) {
                    step_impl_ = &Model::step_with_scheduler;
                }

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

        Model& device(bool use_cuda = true)
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

        void zero_grad() {
            if (optimizer_) {
                optimizer_->zero_grad();
            } else {
                torch::nn::Module::zero_grad();
            }
        }

        void step() { (this->*step_impl_)(); }

        [[nodiscard]] bool has_optimizer() const noexcept { return static_cast<bool>(optimizer_); }

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

            if (!has_optimizer()) {
                throw std::logic_error("Cannot train without an optimizer.");
            }
            if (!has_loss()) {
                throw std::logic_error("Cannot train without a loss function.");
            }
            if (dataset.empty()) {
                return;
            }

            torch::nn::Module::train();
            this->to(device_);

            if constexpr (Config::buffer_vram) {
                if (!device_.is_cuda()) {
                    throw std::runtime_error("VRAM buffering requires the model to be on a CUDA device.");
                }
            }

            auto working_set = TrainingDetails::prepare_dataset<Config>(std::move(dataset), device_);
            auto rng = std::mt19937{std::random_device{}()};

            for (std::size_t epoch = 0; epoch < Config::epochs; ++epoch) {
                TrainingDetails::maybe_shuffle<Config>(working_set, rng);
                TrainingDetails::run_epoch<Config>(*this, working_set);
            }
        }

    private:

        template <bool ShouldShuffle>
    struct ShufflePolicy {
            template <class Dataset, class RNG>
            static void apply(Dataset& dataset, RNG& rng) {
                if constexpr (ShouldShuffle) {
                    if (dataset.size() > 1) {
                        std::shuffle(dataset.begin(), dataset.end(), rng);
                    }
                }
            }
        };

        template <bool BufferVRAM>
        struct VRAMBufferPolicy {
            template <class Sample>
            static void prepare(Sample& sample, const torch::Device& device) {
                if constexpr (BufferVRAM) {
                    sample.first = sample.first.to(device);
                    sample.second = sample.second.to(device);
                }
            }

            [[nodiscard]] static torch::Tensor to_device(const torch::Tensor& tensor, const torch::Device& device) {
                if constexpr (BufferVRAM) {
                    return tensor;
                } else {
                    return tensor.to(device);
                }
            }
        };

        struct TrainingDetails {
            template <class Config, class Dataset>
            static Dataset prepare_dataset(Dataset dataset, const torch::Device& device) {
                for (auto& sample : dataset) {
                    VRAMBufferPolicy<Config::buffer_vram>::prepare(sample, device);
                }
                return dataset;
            }

            template <class Config, class Dataset, class RNG>
            static void maybe_shuffle(Dataset& dataset, RNG& rng) {
                ShufflePolicy<Config::shuffle>::apply(dataset, rng);
            }

            template <class Config, class Dataset>
            static void run_epoch(Model& model, Dataset& dataset) {
                const auto& device = model.device();
                const auto total_samples = dataset.size();
                const auto batch_size = Config::batch_size;

                for (std::size_t offset = 0; offset < total_samples; offset += batch_size) {
                    const auto batch_end = std::min(offset + batch_size, total_samples);
                    if (batch_end <= offset) {
                        continue;
                    }

                    std::vector<torch::Tensor> batch_inputs;
                    std::vector<torch::Tensor> batch_targets;
                    batch_inputs.reserve(batch_end - offset);
                    batch_targets.reserve(batch_end - offset);

                    for (std::size_t index = offset; index < batch_end; ++index) {
                        const auto& sample = dataset[index];
                        batch_inputs.push_back(VRAMBufferPolicy<Config::buffer_vram>::to_device(sample.first, device));
                        batch_targets.push_back(VRAMBufferPolicy<Config::buffer_vram>::to_device(sample.second, device));
                    }

                    if (batch_inputs.empty() || batch_targets.empty()) {
                        continue;
                    }

                    auto inputs = torch::stack(batch_inputs);
                    auto targets = torch::stack(batch_targets);

                    if (inputs.device() != device) {
                        inputs = inputs.to(device);
                    }
                    if (targets.device() != device) {
                        targets = targets.to(device);
                    }

                    model.zero_grad();
                    auto prediction = model.forward(inputs);

                    if (!prediction.sizes().equals(targets.sizes())) {
                        if (targets.numel() == prediction.numel()) {
                            targets = targets.reshape_as(prediction);
                        }
                    }

                    auto loss = model.compute_loss(prediction, targets);
                    loss.backward();
                    model.step();
                }
            }
        };

        std::vector<Layer::Details::RegisteredLayer> layers_{};
        std::unique_ptr<torch::optim::Optimizer> optimizer_{};
        std::unique_ptr<LrScheduler::Details::Scheduler> scheduler_{};
        using StepImpl = void (Model::*)();
        StepImpl step_impl_{&Model::step_not_configured};
        using LossDescriptor = std::variant<Loss::MSEDescriptor, Loss::CrossEntropyDescriptor>;
        std::optional<LossDescriptor> loss_descriptor_{};
        torch::Device device_{torch::kCPU, 0};



        void configure_step_impl() noexcept {
            if (!optimizer_) {
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
                scheduler_->step();
            }
            optimizer_->step();
        }
    };
}

#endif //THOT_CORE_HPP
