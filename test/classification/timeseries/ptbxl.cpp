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
#include <random>
#include <optional>

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

    struct StratifiedFold {
        torch::Tensor train_indices;
        torch::Tensor val_indices;
    };
    std::vector<StratifiedFold> make_stratified_kfold(const torch::Tensor& labels, std::int64_t n_splits, bool shuffle = true, unsigned long long seed = 42ULL) {
        TORCH_CHECK(labels.dim() == 1, "labels must be a 1D tensor");

        // Work on CPU, int64 labels
        auto labels_cpu = labels.to(torch::kCPU, torch::kLong);
        const int64_t N = labels_cpu.size(0);

        // Copy labels to a std::vector<int64_t>
        std::vector<int64_t> labels_vec(N);
        const auto* lbl_ptr = labels_cpu.data_ptr<int64_t>();
        for (int64_t i = 0; i < N; ++i) {
            labels_vec[i] = lbl_ptr[i];
        }

        // Compute unique class values manually
        std::vector<int64_t> classes = labels_vec;
        std::sort(classes.begin(), classes.end());
        classes.erase(std::unique(classes.begin(), classes.end()), classes.end());
        const int64_t n_classes = static_cast<int64_t>(classes.size());

        // For each fold, we keep the list of indices assigned to that fold
        std::vector<std::vector<int64_t>> fold_indices(n_splits);

        std::mt19937_64 rng(seed);

        for (int64_t ci = 0; ci < n_classes; ++ci) {
            const int64_t cls = classes[ci];

            // Collect indices belonging to this class
            std::vector<int64_t> class_idx;
            class_idx.reserve(N);
            for (int64_t i = 0; i < N; ++i) {
                if (labels_vec[i] == cls) {
                    class_idx.push_back(i);
                }
            }

            if (shuffle) {
                std::shuffle(class_idx.begin(), class_idx.end(), rng);
            }

            // Round-robin assign this class' samples to folds
            for (std::size_t j = 0; j < class_idx.size(); ++j) {
                const std::size_t fold_id = j % static_cast<std::size_t>(n_splits);
                fold_indices[fold_id].push_back(class_idx[j]);
            }
        }

        // Build train/val tensors for each fold
        std::vector<StratifiedFold> folds;
        folds.reserve(n_splits);

        for (int64_t k = 0; k < n_splits; ++k) {
            const auto& val_vec = fold_indices[k];

            // Mark validation indices
            std::vector<char> is_val(N, 0);
            for (auto idx : val_vec) {
                is_val[idx] = 1;
            }

            // Everything else goes to train
            std::vector<int64_t> train_vec;
            train_vec.reserve(N - static_cast<int64_t>(val_vec.size()));
            for (int64_t idx = 0; idx < N; ++idx) {
                if (!is_val[idx]) {
                    train_vec.push_back(idx);
                }
            }

            auto train_idx_tensor = torch::tensor(train_vec, torch::TensorOptions().dtype(torch::kLong));
            auto val_idx_tensor   = torch::tensor(val_vec,   torch::TensorOptions().dtype(torch::kLong));

            folds.push_back({std::move(train_idx_tensor), std::move(val_idx_tensor)});
        }

        return folds;
    }


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

int __main() {
    Thot::Model model("PTBXL_ECG");
    const bool use_cuda = torch::cuda::is_available();

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

    const int64_t input_features = train_signals.size(3);
    const int64_t leads = train_signals.size(1) > 1 ? train_signals.size(1) : 12;
    if (input_features <= 0) {
        std::cerr << "Dataset signals must have positive length and feature size." << std::endl;
        return 1;
    }

    const int64_t batch_size = 32;
    const int64_t epochs = 20;
    constexpr int64_t kFolds = 5;
    const int64_t latent_dim = leads * 2; // Vm + W per lead
    const int64_t param_dim = 4;
    const int64_t num_classes = dataset.train.labels.max().item<int64_t>() + 1;
    Thot::Data::Check::Size(train_signals, "Signals");
    Thot::Data::Check::Size(dataset.train.labels, "Label dims");

    std::unordered_set<long> seen;

    for (int64_t i = 0; i < dataset.validation.labels.size(0); ++i) {
        long label = dataset.train.labels[i].item<long>();
        if (seen.insert(label).second)
            std::cout << label << std::endl;
    }


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

    model.use_cuda(use_cuda);
    const auto device = model.device();

    auto folds = make_stratified_kfold(dataset.train.labels, kFolds, /*shuffle=*/true, /*seed=*/42);

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
            const int64_t end   = std::min(start + batch_size, inputs.size(0));

            auto pinned_inputs  = async_pinn_memory(inputs.index({torch::indexing::Slice(start, end)}).contiguous());
            auto pinned_targets = async_pinn_memory(targets.index({torch::indexing::Slice(start, end)}).contiguous());
            auto host_inputs    = pinned_inputs.materialize();
            auto host_targets   = pinned_targets.materialize();

            auto device_inputs  = to_device(host_inputs);
            auto device_targets = to_device(host_targets).to(torch::kLong);

            auto artifacts      = forward_physics_informed(model, device_inputs, latent_dim, param_dim, num_classes, leads);

            auto cls_loss       = torch::nn::functional::cross_entropy(
                artifacts.logits,
                device_targets,
                torch::nn::functional::CrossEntropyFuncOptions().label_smoothing(0.05)
            );
            auto physics_loss   = artifacts.physics_residual.pow(2).mean();
            auto param_sparsity = artifacts.inferred_params.abs().mean();
            auto loss           = cls_loss + 0.1 * physics_loss + 0.01 * param_sparsity;

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

    for (int64_t fold = 0; fold < kFolds; ++fold) {
        const auto& f = folds[fold];
        std::cout << "\n========== Fold " << (fold + 1) << "/" << kFolds << " ==========\n";

        auto x_train = train_signals.index_select(0, f.train_indices);
        auto y_train = dataset.train.labels.index_select(0, f.train_indices);

        auto x_val   = train_signals.index_select(0, f.val_indices);
        auto y_val   = dataset.train.labels.index_select(0, f.val_indices);

        Thot::Data::Check::Size(x_train, "Fold train signals");
        Thot::Data::Check::Size(y_train, "Fold train labels");
        Thot::Data::Check::Size(x_val,   "Fold val signals");
        Thot::Data::Check::Size(y_val,   "Fold val labels");

        for (int64_t epoch = 0; epoch < epochs; ++epoch) {
            const auto train_loss = run_epoch(x_train, y_train, true);
            const auto val_loss   = run_epoch(x_val,   y_val,   false);

            std::cout << "[Fold " << (fold + 1) << "/" << kFolds
                      << " | Epoch " << (epoch + 1) << "/" << epochs
                      << "] train=" << train_loss
                      << " val="   << val_loss  << std::endl;
        }
        /*
        std::cout << "\nFold " << (fold + 1) << " evaluation..." << std::endl;
        model.evaluate(
            x_val, y_val,
            Thot::Evaluation::Classification,
            {
                Thot::Metric::Classification::Accuracy,
                Thot::Metric::Classification::F1,
                Thot::Metric::Classification::Precision,
                Thot::Metric::Classification::Recall,
                Thot::Metric::Classification::AUCROC,
            },
            {.batch_size = static_cast<std::size_t>(batch_size), .buffer_vram = 1, .print_per_class=false}
        ); std::cout << "\n" << std::endl;
        */
    }

    std::vector<Thot::Metric::Classification::Descriptor> metrics;


    std::cout << "\nRunning evaluation..." << std::endl;
    model.evaluate(validation_signals, dataset.validation.labels, Thot::Evaluation::Classification, {
        Thot::Metric::Classification::Accuracy,
        Thot::Metric::Classification::F1,
        Thot::Metric::Classification::Precision,
        Thot::Metric::Classification::Recall,
        Thot::Metric::Classification::AUCROC,
        }, {.batch_size=static_cast<std::size_t>(batch_size), .buffer_vram=1});

    return 0;
}

