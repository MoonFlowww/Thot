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
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "../include/Thot.h"


struct ECGDataset {
    torch::Tensor signals;
    torch::Tensor labels;
};

struct ECGDatasetSplit {
    ECGDataset train;
    ECGDataset validation;
};


ECGDatasetSplit load_ptbxl_dataset(const std::string& root, bool low_res, float train_split) {
    auto [train_inputs, train_targets, validation_inputs, validation_targets] = Thot::Data::Load::PTBXL<>(root, low_res, train_split, true, true, 0.1f);
    return {{std::move(train_inputs), std::move(train_targets)}, {std::move(validation_inputs), std::move(validation_targets)}};
}


int main()
{
    Thot::Model model("PTBXL_ECG");
    constexpr bool load_existing_model = false;
    const bool use_cuda = torch::cuda::is_available();
    std::cout << "Cuda: " << use_cuda << std::endl;
    model.to_device(use_cuda);

    const std::string dataset_root = "/home/moonfloww/Projects/DATASETS/ECG_ACC";
    const auto dataset = load_ptbxl_dataset(dataset_root, true, 0.8f);


    auto train_signals = dataset.train.signals.transpose(1, 2).contiguous();
    auto validation_signals = dataset.validation.signals.transpose(1, 2).contiguous();

    const int64_t input_features = train_signals.size(2);
    const int64_t sequence_length = train_signals.size(1);
    if (input_features <= 0 || sequence_length <= 0) {
        std::cerr << "Dataset signals must have positive length and feature size." << std::endl;
        return 1;
    }

    const int64_t batch_size = 64;
    const int64_t epochs = 40;
    const int64_t steps_per_epoch = std::max<int64_t>(1, (train_signals.size(0) + batch_size - 1) / batch_size);

    if (!load_existing_model) {
        model.add(Thot::Layer::LSTM({
                              .input_size = input_features,
                              .hidden_size = 128,
                              .num_layers = 2,
                              .dropout = 0.1,
                              .batch_first = true,
                              .bidirectional = true
                          }, Thot::Activation::Identity, Thot::Initialization::XavierUniform), "lstm");
        
        model.add(Thot::Layer::Flatten(), "flat");
        model.add(Thot::Layer::HardDropout({.probability = 0.5}), "HD1");
        model.add(Thot::Layer::FC({sequence_length * 128 * 2, 256, true}, Thot::Activation::SiLU, Thot::Initialization::HeNormal), "fc1"); // seq_len * Hsize LSTM * bidirectional
        model.add(Thot::Layer::HardDropout({.probability = 0.5}), "HD2");
        model.add(Thot::Layer::FC({256, 128, true}, Thot::Activation::SiLU, Thot::Initialization::HeNormal), "fc2");
        model.add(Thot::Layer::FC({128, 5, true}, Thot::Activation::Identity, Thot::Initialization::HeNormal), "end");

        model.links({
            Thot::LinkSpec{Thot::Port::parse("@input"), Thot::Port::parse("lstm")},
            Thot::LinkSpec{Thot::Port::parse("lstm"), Thot::Port::parse("flatten")},

            Thot::LinkSpec{Thot::Port::parse("flat"), Thot::Port::parse("dropout1")},

            Thot::LinkSpec{Thot::Port::parse("HD1"), Thot::Port::parse("fc1")},
            Thot::LinkSpec{Thot::Port::parse("fc1"), Thot::Port::parse("HD2")},
            Thot::LinkSpec{Thot::Port::parse("dropout2"), Thot::Port::parse("fc2")},
            Thot::LinkSpec{Thot::Port::parse("fc2"), Thot::Port::parse("classifier")},
            Thot::LinkSpec{Thot::Port::parse("end"), Thot::Port::parse("@output")}
        }, true);

        model.set_optimizer(
        Thot::Optimizer::AdamW({.learning_rate = 2e-4, .weight_decay = 1e-4}),
        Thot::LrScheduler::CosineAnnealing({
            .T_max = static_cast<std::size_t>(epochs) * static_cast<std::size_t>(steps_per_epoch),
            .eta_min = 1e-6,
            .warmup_steps = static_cast<std::size_t>(steps_per_epoch),
            .warmup_start_factor = 0.1
        }));

        model.set_loss(Thot::Loss::BCEWithLogits());
    }

    Thot::TrainOptions train_options{};
    train_options.epoch = static_cast<std::size_t>(epochs);
    train_options.batch_size = static_cast<std::size_t>(batch_size);
    train_options.shuffle = true;
    train_options.buffer_vram = 0;
    train_options.graph_mode = Thot::GraphMode::Capture;
    train_options.restore_best_state = true;
    train_options.enable_amp = true;
    train_options.memory_format = torch::MemoryFormat::Contiguous;
    train_options.test = std::make_pair(validation_signals, dataset.validation.labels);

    if (!load_existing_model)
        model.train(train_signals, dataset.train.labels, {
            .epoch = static_cast<std::size_t>(epochs),
            .batch_size = static_cast<std::size_t>(batch_size),
            .shuffle = true,
            .buffer_vram = 0,
            .restore_best_state = true,
            .test = std::make_pair(validation_signals, dataset.validation.labels),
            .graph_mode = Thot::GraphMode::Capture,
            .enable_amp = true,
            .memory_format = torch::MemoryFormat::Contiguous,
        });
    

    if (dataset.validation.signals.size(0) > 0) {
        model.evaluate(validation_signals, dataset.validation.labels, Thot::Evaluation::Classification,
        {
            Thot::Metric::Classification::Accuracy,
            Thot::Metric::Classification::Precision,
            Thot::Metric::Classification::Recall,
            Thot::Metric::Classification::F1
        }, {.batch_size = static_cast<std::size_t>(batch_size)});
    }

    return 0;
}