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

#include "../../../include/Thot.h"


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


int _main() {
    Thot::Model model("PTBXL_ECG");
    const bool use_cuda = torch::cuda::is_available();
    std::cout << "Cuda: " << use_cuda << std::endl;
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
    const int64_t sequence_length = train_signals.size(1);
    if (input_features <= 0 || sequence_length <= 0) {
        std::cerr << "Dataset signals must have positive length and feature size." << std::endl;
        return 1;
    }

    const int64_t batch_size = 64;
    const int64_t epochs = 20;
    const int64_t steps_per_epoch = std::max<int64_t>(1, (train_signals.size(0) + batch_size - 1) / batch_size);

    Thot::Data::Check::Size(train_signals);


    model.add(Thot::Layer::Conv2d({
            .in_channels= 12,.out_channels= 32, .kernel_size= {3, 3}, .stride= {1, 1}, .padding= {1, 1}, .dilation= {1, 1}, .groups= 1, .bias= false},
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
        { .input_size   = 1000, .hidden_size  = 128, .num_layers= 5, .dropout= 0.15, .batch_first= true, .bidirectional=true },
        Thot::Activation::Identity, Thot::Initialization::XavierUniform), "lstm");

    model.add(Thot::Layer::Reduce({ .op= Thot::Layer::ReduceOp::Mean, .dims= {1}, .keep_dim = false }), "Reduc");

    model.add(Thot::Layer::SoftDropout({ .probability = 0.5 }), "SD1");
    model.add(Thot::Layer::FC({256, 128, true}, Thot::Activation::SiLU, Thot::Initialization::HeNormal), "fc1");

    model.add(Thot::Layer::HardDropout({ .probability = 0.5 }), "HD1");
    model.add(Thot::Layer::FC({128, 5, true}, Thot::Activation::Identity, Thot::Initialization::HeNormal), "end");


    model.set_optimizer(
    Thot::Optimizer::AdamW({.learning_rate = 2e-4, .weight_decay = 1e-4}),
    Thot::LrScheduler::CosineAnnealing({
        .T_max = static_cast<std::size_t>(epochs) * static_cast<std::size_t>(steps_per_epoch),
        .eta_min = 1e-6,
        .warmup_steps = static_cast<std::size_t>(steps_per_epoch),
        .warmup_start_factor = 0.1
    }));

    model.set_loss(Thot::Loss::CrossEntropy({.label_smoothing = 0.05f}));

    Thot::Data::Normalization::Zscore(train_signals, {.lag=30, .forward_only=true});
    Thot::Data::Normalization::Zscore(validation_signals, {.lag=30, .forward_only=true});
    Thot::Data::Check::Size(train_signals, "Zscore");

    model.train(train_signals, dataset.train.labels, {
        .epoch = static_cast<std::size_t>(epochs),
        .batch_size = static_cast<std::size_t>(batch_size),
        .shuffle = true,
        .buffer_vram = 0,
        .restore_best_state = true,
        .test = std::vector<at::Tensor>{validation_signals, dataset.validation.labels},
        .enable_amp = true,
        .memory_format = torch::MemoryFormat::Contiguous,
    });
    

    if (dataset.validation.signals.size(0) > 0) {
        model.evaluate(validation_signals, dataset.validation.labels, Thot::Evaluation::Classification,
        {
            Thot::Metric::Classification::Accuracy,
            Thot::Metric::Classification::Precision,
            Thot::Metric::Classification::Recall,
            Thot::Metric::Classification::FalsePositiveRate,
            Thot::Metric::Classification::FalseNegativeRate,
            Thot::Metric::Classification::F1,
            Thot::Metric::Classification::AUCROC
        }, {.batch_size = static_cast<std::size_t>(batch_size)});
    }

    return 0;
}

