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

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "activation/activation.hpp"
#include "activation/details/relu.hpp"
#include "initialization/initialization.hpp"
#include "layer/layer.hpp"
#include "loss/loss.hpp"
#include "loss/details/mse.hpp"
#include "optimizer/optimizer.hpp"
#include "optimizer/details/sgd.hpp"

namespace Thot {

class Model : public torch::nn::Module {
public:
    Model() = default;

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

private:
    struct DenseLayer {
        torch::nn::Linear linear{nullptr};
        Activation::Type activation{Activation::Type::Identity};
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

}  // namespace Thot

#endif //THOT_CORE_HPP
