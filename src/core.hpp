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
#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "activation/activation.hpp"
#include "activation/apply.hpp"
#include "initialization/initialization.hpp"
#include "layer/layer.hpp"
#include "loss/loss.hpp"
#include "loss/details/mse.hpp"
#include "optimizer/optimizer.hpp"
#include "optimizer/details/sgd.hpp"

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

        inline constexpr auto kDefaultTrainingConfig = TrainingConfig<10, 32, true, false>{};
    }

    class Model : public torch::nn::Module {
    public:
        Model() = default;

        using torch::nn::Module::train;

        void add(const Layer::FCDescriptor& descriptor) {
            if (descriptor.options.in_features <= 0 || descriptor.options.out_features <= 0) {
                throw std::invalid_argument("Fully connected layers require positive in/out features.");
            }
            const auto index = dense_layers_.size();
            auto options = torch::nn::LinearOptions(descriptor.options.in_features,
                                                    descriptor.options.out_features)
                                .bias(descriptor.options.bias);
            auto layer = register_module("fc_" + std::to_string(index), torch::nn::Linear(options));
            apply_initialization(layer, descriptor);
            dense_layers_.push_back(DenseLayer{layer, descriptor.activation.type});
        }

        void set_optimizer(const Optimizer::SGDDescriptor& descriptor) {
            if (dense_layers_.empty()) {
                throw std::logic_error("Cannot create optimizer before any layer has been registered.");
            }
            auto options = Optimizer::Details::to_torch_options(descriptor.options);
            optimizer_ = std::make_unique<torch::optim::SGD>(this->parameters(), options);
        }

        void set_loss(const Loss::MSEDescriptor& descriptor) {
            loss_descriptor_ = descriptor;
        }

        [[nodiscard]] torch::Tensor forward(torch::Tensor input) {
            auto output = std::move(input);
            for (auto& layer : dense_layers_) {
                output = layer.linear->forward(output);
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
            return Loss::Details::compute(*loss_descriptor_, prediction, target, weight);
        }

        void zero_grad() {
            if (optimizer_) {
                optimizer_->zero_grad();
            } else {
                torch::nn::Module::zero_grad();
            }
        }

        void step() {
            if (!optimizer_) {
                throw std::logic_error("Optimizer has not been configured.");
            }
            optimizer_->step();
        }

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

            const auto device = typename Config::DevicePolicy::select();
            torch::nn::Module::train();
            this->to(device);

            auto working_set = TrainingDetails::prepare_dataset<Config>(std::move(dataset), device);
            auto rng = std::mt19937{std::random_device{}()};

            for (std::size_t epoch = 0; epoch < Config::epochs; ++epoch) {
                TrainingDetails::maybe_shuffle<Config>(working_set, rng);
                TrainingDetails::run_epoch<Config>(*this, working_set, device);
            }
        }


    private:
        struct DenseLayer {
            torch::nn::Linear linear{nullptr};
            Activation::Type activation{Activation::Type::Identity};
        };

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
            static void run_epoch(Model& model, Dataset& dataset, const torch::Device& device) {
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

                    model.zero_grad();
                    auto prediction = model.forward(inputs);
                    auto loss = model.compute_loss(prediction, targets);
                    loss.backward();
                    model.step();
                }
            }
        };

        static void apply_initialization(const torch::nn::Linear& layer, const Layer::FCDescriptor& descriptor) {
            switch (descriptor.initialization.type) {
                case Initialization::Type::XavierNormal:
                    torch::nn::init::xavier_normal_(layer->weight);
                    if (descriptor.options.bias && layer->bias.defined()) {
                        torch::nn::init::zeros_(layer->bias);
                    }
                    break;
                case Initialization::Type::Default:
                default:
                    break;
            }
        }

        std::vector<DenseLayer> dense_layers_{};
        std::unique_ptr<torch::optim::Optimizer> optimizer_{};
        std::optional<Loss::MSEDescriptor> loss_descriptor_{};
    };
}

#endif //THOT_CORE_HPP
