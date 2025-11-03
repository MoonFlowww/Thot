#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "../include/Thot.h"

namespace {
    struct ECGDataset {
        torch::Tensor signals;
        torch::Tensor labels;
    };

    struct ECGDatasetSplit {
        ECGDataset train;
        ECGDataset validation;
    };

}

ECGDatasetSplit load_ptbxl_dataset(const std::string& root, bool low_res, float train_split)
{
    auto [train_inputs, train_targets, validation_inputs, validation_targets] = Thot::Data::Load::PTBXL<>(root, low_res, train_split, true);

    if (train_inputs.dim() != 3 || train_inputs.size(1) != 12) {
        throw std::runtime_error("Expected PTB-XL training signals to have shape [N, 12, L].");
    }
    if (train_targets.size(0) != train_inputs.size(0)) {
        throw std::runtime_error("PTB-XL training inputs and targets size mismatch");
    }
    if (train_targets.dtype() != torch::kInt64) {
        throw std::runtime_error("PTB-XL training labels must be int64_t");
    }

    const auto label_min = train_targets.min().item<std::int64_t>();
    const auto label_max = train_targets.max().item<std::int64_t>();
    if (label_min < 0 || label_max > 4) {
        throw std::runtime_error("PTB-XL labels must fall within the five diagnostic superclasses");
    }
    if (validation_inputs.defined() && validation_inputs.size(0) > 0) {
        if (validation_inputs.dim() != 3 || validation_inputs.size(1) != 12) {
            throw std::runtime_error("Expected PTB-XL validation signals to have shape [N, 12, L].");
        }
        if (validation_targets.size(0) != validation_inputs.size(0)) {
            throw std::runtime_error("PTB-XL validation inputs and targets size mismatch");
        }
    }
    return {{std::move(train_inputs), std::move(train_targets)},
            {std::move(validation_inputs), std::move(validation_targets)}};

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

    const std::int64_t batch_size = 64;
    const std::int64_t epochs = 40;
    const std::int64_t num_classes = 5;
    const std::int64_t steps_per_epoch = std::max<std::int64_t>(1, (dataset.train.signals.size(0) + batch_size - 1) / batch_size);

    if (!load_existing_model) {
        model.add(Thot::Block::Sequential({
            Thot::Layer::Conv1d({12, 64, {7}, {2}, {3}}, Thot::Activation::SiLU, Thot::Initialization::HeNormal),
                      Thot::Layer::Conv1d({64, 64, {5}, {1}, {2}}, Thot::Activation::SiLU, Thot::Initialization::HeNormal),
                      Thot::Layer::MaxPool1d({{2}, {2}})
                  }),
                  "stem");

        model.add(Thot::Block::Sequential({
            Thot::Layer::Conv1d({64, 128, {5}, {2}, {2}}, Thot::Activation::SiLU, Thot::Initialization::HeNormal),
                      Thot::Layer::Conv1d({128, 128, {3}, {1}, {1}}, Thot::Activation::SiLU, Thot::Initialization::HeNormal),
                      Thot::Layer::MaxPool1d({{2}, {2}})
                  }),
                  "features1");

        model.add(Thot::Block::Sequential({
            Thot::Layer::Conv1d({128, 256, {3}, {1}, {1}}, Thot::Activation::SiLU, Thot::Initialization::HeNormal),
                      Thot::Layer::Conv1d({256, 256, {3}, {1}, {1}}, Thot::Activation::SiLU, Thot::Initialization::HeNormal),
                      Thot::Layer::MaxPool1d({{2}, {2}})
                  }),
                  "features2");
        model.add(Thot::Block::Sequential({
                      Thot::Layer::Conv1d({256, 256, {3}, {1}, {1}}, Thot::Activation::SiLU, Thot::Initialization::HeNormal),
                      Thot::Layer::AdaptiveAvgPool1d({{1}})
                  }),
                  "head");
        model.add(Thot::Layer::Flatten(), "flatten");
        model.add(Thot::Layer::FC({256, 128, true}, Thot::Activation::SiLU, Thot::Initialization::HeNormal), "fc1");
        model.add(Thot::Layer::HardDropout({.probability = 0.5}), "dropout");
        model.add(Thot::Layer::FC({128, num_classes, true}, Thot::Activation::Identity, Thot::Initialization::HeNormal), "classifier");

        model.links({
            Thot::LinkSpec{Thot::Port::parse("@input"), Thot::Port::parse("stem")},
            Thot::LinkSpec{Thot::Port::parse("stem"), Thot::Port::parse("features1")},
            Thot::LinkSpec{Thot::Port::parse("features1"), Thot::Port::parse("features2")},
            Thot::LinkSpec{Thot::Port::parse("features2"), Thot::Port::parse("head")},
            Thot::LinkSpec{Thot::Port::parse("head"), Thot::Port::parse("flatten")},
            Thot::LinkSpec{Thot::Port::parse("flatten"), Thot::Port::parse("fc1")},
            Thot::LinkSpec{Thot::Port::parse("fc1"), Thot::Port::parse("dropout")},
            Thot::LinkSpec{Thot::Port::parse("dropout"), Thot::Port::parse("classifier")},
            Thot::LinkSpec{Thot::Port::parse("classifier"), Thot::Port::parse("@output")}
        }, true);

        model.set_optimizer(
        Thot::Optimizer::AdamW({.learning_rate = 2e-4, .weight_decay = 1e-4}),
        Thot::LrScheduler::CosineAnnealing({
            .T_max = static_cast<std::size_t>(epochs) * static_cast<std::size_t>(steps_per_epoch),
            .eta_min = 1e-6,
            .warmup_steps = static_cast<std::size_t>(steps_per_epoch),
            .warmup_start_factor = 0.1
        }));

        model.set_loss(Thot::Loss::CrossEntropy({.label_smoothing = 0.01f}));
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
    train_options.test = std::make_pair(dataset.validation.signals, dataset.validation.labels);

    if (!load_existing_model) {
        model.train(dataset.train.signals, dataset.train.labels, train_options);
    }

    if (dataset.validation.signals.size(0) > 0) {
        model.evaluate(dataset.validation.signals,
                       dataset.validation.labels,
                       Thot::Evaluation::Classification,
                       {
                           Thot::Metric::Classification::Accuracy,
                           Thot::Metric::Classification::Precision,
                           Thot::Metric::Classification::Recall,
                           Thot::Metric::Classification::F1
                       },
                       {.batch_size = static_cast<std::size_t>(batch_size)});
    }

    return 0;
}