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


int main()
{
    Thot::Model model("PTBXL_ECG");
    constexpr bool load_existing_model = false;
    const bool use_cuda = torch::cuda::is_available();
    std::cout << "Cuda: " << use_cuda << std::endl;
    model.use_cuda(use_cuda);

    const auto dataset = load_ptbxl_dataset("/home/moonfloww/Projects/DATASETS/ECG_ACC", true, 0.8f);


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

    Thot::Data::Check::Size(train_signals);


    model.add(Thot::Layer::xLSTM({ .input_size = input_features, .hidden_size = 128, .num_layers = 2, .dropout = 0.1, .batch_first = true, .bidirectional = true }, Thot::Activation::Identity, Thot::Initialization::XavierUniform), "lstm");

    model.add(Thot::Layer::Reduce({.op=Thot::Layer::ReduceOp::Max, .dims = {1}, .keep_dim=false}), "Reduc");

    model.add(Thot::Layer::SoftDropout({ .probability = 0.5}), "SD1");
    model.add(Thot::Layer::FC({ 256, 128, true }, Thot::Activation::SiLU, Thot::Initialization::HeNormal), "fc1");
    model.add(Thot::Layer::HardDropout({ .probability = 0.5 }), "HD1");
    model.add(Thot::Layer::FC({ 128, 5, true }, Thot::Activation::Identity, Thot::Initialization::HeNormal), "end");


    model.links({
        {Thot::Port::Module("@input"), Thot::Port::Module("lstm")},
        {Thot::Port::Module("lstm"),   Thot::Port::Module("Reduc")},
        {Thot::Port::Module("Reduc"),  Thot::Port::Module("SD1")},
        {Thot::Port::Module("SD1"),    Thot::Port::Module("fc1")},
        {Thot::Port::Module("fc1"),    Thot::Port::Module("HD1")},
        {Thot::Port::Module("HD1"),    Thot::Port::Module("end")},
        {Thot::Port::Module("end"),    Thot::Port::Module("@output")}
    }, {.enable_graph_capture = true});


    model.set_optimizer(
    Thot::Optimizer::AdamW({.learning_rate = 2e-4, .weight_decay = 1e-4}),
    Thot::LrScheduler::CosineAnnealing({
        .T_max = static_cast<std::size_t>(epochs) * static_cast<std::size_t>(steps_per_epoch),
        .eta_min = 1e-6,
        .warmup_steps = static_cast<std::size_t>(steps_per_epoch),
        .warmup_start_factor = 0.1
    }));

    model.set_loss(Thot::Loss::CrossEntropy({.label_smoothing = 0.05f}));



    model.train(train_signals, dataset.train.labels, {
        .epoch = static_cast<std::size_t>(epochs),
        .batch_size = static_cast<std::size_t>(batch_size),
        .shuffle = true,
        .buffer_vram = 0,
        .restore_best_state = true,
        .test = std::vector<at::Tensor>{validation_signals, dataset.validation.labels},
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
