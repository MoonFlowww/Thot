#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#include <torch/torch.h>
#include <torch/nn/functional.h>

#include "../../../include/Thot.h"

inline Thot::AsyncPinnedTensor async_pinn_memory(torch::Tensor tensor) {
    return Thot::async_pin_memory(std::move(tensor));
}
namespace {
    struct ECGDataset {
        torch::Tensor signals;
        torch::Tensor labels;
    };

    struct ECGDatasetSplit {
        ECGDataset train;
        ECGDataset validation;
    };


    ECGDatasetSplit load_ptbxl_dataset(const std::string& root, bool low_res, float train_split) {
        auto [train_inputs, train_targets, validation_inputs, validation_targets] = Thot::Data::Load::PTBXL<>(root, low_res, train_split, true, false);
        return {{std::move(train_inputs), std::move(train_targets)}, {std::move(validation_inputs), std::move(validation_targets)}};
    }

    // Simple surrogate of electrophysiology ODEs: we treat each lead as a low-dimensional
    // pair (Vm, W) and enforce that spatial conduction, recovery, dispersion and leak terms
    // balance each other. Residuals close to zero indicate physically plausible latent
    // reconstructions from the ECG.
    torch::Tensor compute_ep_residuals(const torch::Tensor& latent, const torch::Tensor& params, int leads) {
        const int64_t batch = latent.size(0);
        const int64_t latent_per_lead = latent.size(1) / leads;
        const auto conduction = torch::softplus(params.select(1, 0)).view({batch, 1, 1});
        const auto recovery = torch::softplus(params.select(1, 1)).view({batch, 1, 1});
        const auto dispersion = torch::sigmoid(params.select(1, 2)).view({batch, 1, 1});
        const auto leak = torch::sigmoid(params.select(1, 3)).view({batch, 1, 1});

        auto reshaped = latent.view({batch, leads, latent_per_lead});
        auto Vm = reshaped.slice(/*dim=*/2, 0, 1);
        auto W = reshaped.slice(/*dim=*/2, 1, 2);

        auto shifted_left = torch::roll(Vm, 1, /*dim=*/1);
        auto shifted_right = torch::roll(Vm, -1, /*dim=*/1);
        auto laplacian = shifted_left - 2 * Vm + shifted_right;

        auto activation_drive = conduction * laplacian;
        auto recovery_drive = recovery * (Vm - W);
        auto dispersion_term = dispersion * torch::tanh(Vm);
        auto leak_term = leak * Vm;

        auto residual = activation_drive - recovery_drive - dispersion_term - leak_term;
        return residual.squeeze(2);
    }

    struct TrainingArtifacts {
        torch::Tensor logits;
        torch::Tensor latent_states;
        torch::Tensor inferred_params;
        torch::Tensor physics_residual;
        torch::Tensor risk_features;
    };

    TrainingArtifacts forward_physics_informed(Thot::Model& model, torch::Tensor inputs, int latent_dim, int param_dim, int num_classes, int leads) {
        auto combined = model.forward(std::move(inputs));

        const auto splits = combined.split({latent_dim, param_dim, num_classes}, /*dim=*/1);
        auto latent_states = splits[0];
        auto inferred_params = splits[1];
        auto logits = splits[2];

        auto residual = compute_ep_residuals(latent_states, inferred_params, leads);
        auto dispersion = inferred_params.select(1, 2).unsqueeze(1);
        auto repol_slope = torch::softplus(inferred_params.select(1, 1)).unsqueeze(1);
        auto activation_energy = latent_states.view({latent_states.size(0), leads, latent_dim / leads}).mean(2);
        auto stability = residual.abs().mean(1, true);
        auto risk_features = torch::cat({dispersion, repol_slope, activation_energy.mean(1, true), stability}, 1);

        return {std::move(logits), std::move(latent_states), std::move(inferred_params), std::move(residual), std::move(risk_features)};
    }

}

int main() {
    Thot::Model model("PTBXL_ECG");
    const bool use_cuda = torch::cuda::is_available();
    model.use_cuda(use_cuda);

    const auto dataset = load_ptbxl_dataset("/home/moonfloww/Projects/DATASETS/Timeserie/ECG_ACC", true, 0.8f);


    auto prepare_signals = [](torch::Tensor signals) {
        if (!signals.defined())
            return signals;

        if (signals.dim() < 3)
            throw std::runtime_error("PTB-XL signals must be at least 3D (batch, channels, timesteps).");

        auto prepared = signals.contiguous();
        if (prepared.dim() == 3)
            prepared = prepared.unsqueeze(2);
        return prepared;
    };

    auto train_signals = prepare_signals(dataset.train.signals);
    auto validation_signals = prepare_signals(dataset.validation.signals);

    const int64_t input_features = train_signals.size(2);
    const int64_t leads = train_signals.size(3) > 1 ? train_signals.size(3) : 12;
    if (input_features <= 0) {
        std::cerr << "Dataset signals must have positive length and feature size." << std::endl;
        return 1;
    }

    const int64_t batch_size = 32;
    const int64_t epochs = 5;
    const int64_t latent_dim = leads * 2; // Vm + W per lead
    const int64_t param_dim = 4;
    const int64_t num_classes = dataset.train.labels.max().item<int64_t>() + 1;
    Thot::Data::Check::Size(train_signals);


    model.add(Thot::Layer::Conv2d({
            .in_channels= leads,.out_channels= 32, .kernel_size= {3, 3}, .stride= {1, 1}, .padding= {1, 1}, .dilation= {1, 1}, .groups= 1, .bias= false},
            Thot::Activation::SiLU, Thot::Initialization::HeNormal), "conv1");

    model.add(Thot::Layer::Conv2d(
            {.in_channels= 32, .out_channels= 64, .kernel_size= {3, 3}, .stride= {1, 1}, .padding= {2, 1}, .dilation= {2, 1}, .groups= 1, .bias= false},
            Thot::Activation::SiLU, Thot::Initialization::HeNormal), "conv2");

    model.add(Thot::Layer::Conv2d(
        {.in_channels= 64, .out_channels = 64, .kernel_size= {3, 3}, .stride= {2, 1}, .padding= {1, 1}, .dilation= {1, 1}, .groups= 1, .bias= false},
        Thot::Activation::SiLU, Thot::Initialization::HeNormal), "conv3");


    model.add(Thot::Layer::Conv2d(
        {.in_channels= 64, .out_channels= 128, .kernel_size= {3, 3}, .stride= {1, 1}, .padding= {4, 1}, .dilation= {4, 1}, .groups= 1, .bias= false},
            Thot::Activation::SiLU, Thot::Initialization::HeNormal), "conv4");

    model.add(Thot::Layer::Conv2d(
        {.in_channels = 128, .out_channels = 128, .kernel_size  = {3, 3}, .stride = {2, 1}, .padding= {1, 1}, .dilation = {1, 1}, .groups = 1, .bias = false},
        Thot::Activation::SiLU, Thot::Initialization::HeNormal), "conv5");

    model.add(Thot::Layer::Reduce({ .op= Thot::Layer::ReduceOp::Mean, .dims= {2}, .keep_dim = false }), "flat");


    model.add(Thot::Layer::xLSTM(
{ .input_size   = input_features, .hidden_size  = 128, .num_layers= 3, .dropout= 0.1, .batch_first= true, .bidirectional=true },
        Thot::Activation::Identity, Thot::Initialization::XavierUniform), "lstm");

    model.add(Thot::Layer::Reduce({ .op= Thot::Layer::ReduceOp::Mean, .dims= {1}, .keep_dim = false }), "Reduc");

    model.add(Thot::Layer::SoftDropout({ .probability = 0.3}), "SD1");
    model.add(Thot::Layer::FC({256, 128, true}, Thot::Activation::SiLU, Thot::Initialization::HeNormal), "fc1");
    model.add(Thot::Layer::HardDropout({ .probability = 0.2 }), "HD1");
    model.add(Thot::Layer::FC({128, latent_dim + param_dim + num_classes, true}, Thot::Activation::Identity, Thot::Initialization::HeNormal), "heads");


    auto parameters = model.parameters();
    torch::optim::AdamW optimizer(parameters, torch::optim::AdamWOptions(2e-4).weight_decay(1e-4));

    Thot::Data::Transform::Normalization::Zscore(train_signals, {.lag=30, .forward_only=true});
    Thot::Data::Transform::Normalization::Zscore(validation_signals, {.lag=30, .forward_only=true});
    Thot::Data::Check::Size(train_signals, "Zscore");

const auto device = model.device();

    auto to_device = [&device](const torch::Tensor& tensor) {
        if (!tensor.defined()) return tensor;
        const bool non_blocking = device.is_cuda() && tensor.is_pinned();
        return tensor.to(device, tensor.scalar_type(), non_blocking);
    };

    auto run_epoch = [&](const torch::Tensor& inputs, const torch::Tensor& targets, bool training) {
        const int64_t batches = (inputs.size(0) + batch_size - 1) / batch_size;
        double running_loss = 0.0;
        model.train(training);

        for (int64_t i = 0; i < batches; ++i) {
            const int64_t start = i * batch_size;
            const int64_t end = std::min(start + batch_size, inputs.size(0));
            auto batch_inputs = inputs.index({torch::indexing::Slice(start, end)});
            auto batch_targets = targets.index({torch::indexing::Slice(start, end)});

            auto pinned_inputs = async_pinn_memory(batch_inputs.contiguous());
            auto pinned_targets = async_pinn_memory(batch_targets.contiguous());
            auto host_inputs = pinned_inputs.materialize();
            auto host_targets = pinned_targets.materialize();

            auto device_inputs = to_device(host_inputs);
            auto device_targets = to_device(host_targets).to(torch::kLong);

            auto artifacts = forward_physics_informed(model, device_inputs, latent_dim, param_dim, num_classes, leads);
            auto cls_loss = torch::nn::functional::cross_entropy(artifacts.logits, device_targets, torch::nn::functional::CrossEntropyFuncOptions().label_smoothing(0.05));
            auto physics_loss = artifacts.physics_residual.pow(2).mean();
            auto param_sparsity = artifacts.inferred_params.abs().mean();
            auto loss = cls_loss + 0.1 * physics_loss + 0.01 * param_sparsity;

            if (training) {
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();
            }

            if (!training && i == batches - 1) {
                auto risk_marker = artifacts.risk_features.mean().item<double>();
                std::cout << "    risk surrogate mean=" << risk_marker << std::endl;
            }

            running_loss += loss.detach().cpu().item<double>();
        }

        return running_loss / std::max<int64_t>(1, batches);
    };

    for (int64_t epoch = 0; epoch < epochs; ++epoch) {
        const auto train_loss = run_epoch(train_signals, dataset.train.labels, true);
        const auto val_loss = run_epoch(validation_signals, dataset.validation.labels, false);
        std::cout << "[Epoch " << (epoch + 1) << "/" << epochs << "] train=" << train_loss << " val=" << val_loss << std::endl;
    }

    return 0;
}

