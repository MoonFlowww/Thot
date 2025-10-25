#include <iostream>
#include <cstddef>
#include <vector>
#include <torch/torch.h>
#include <utility>
#include "../include/Thot.h"




int main() {
    Thot::Model model;
    std::cout << "Cuda: " << torch::cuda::is_available() << std::endl;
    model.to_device(torch::cuda::is_available());

    model.add(Thot::Block::Sequential({
        Thot::Layer::Conv2d(
            {3, 64, {3,3}, {1,1}, {1,1}, {1,1}, 1, false},
            Thot::Activation::Raw,
            Thot::Initialization::KaimingNormal
        ),
        Thot::Layer::BatchNorm2d({64, 1e-5, 0.1, true, true}, Thot::Activation::GeLU),
        Thot::Layer::Conv2d(
            {64, 64, {3,3}, {1,1}, {1,1}, {1,1}, 1, false}, // bias false under BN
            Thot::Activation::Raw,
            Thot::Initialization::KaimingNormal
        ),
        Thot::Layer::BatchNorm2d({64, 1e-5, 0.1, true, true}, Thot::Activation::GeLU),
        Thot::Layer::MaxPool2d({{2,2}, {2,2}})
    }));

    model.add(Thot::Layer::Dropout({ .probability = 0.3 }));

    model.add(Thot::Block::Residual({
        Thot::Layer::Conv2d(
            {64, 128, {3,3}, {2,2}, {1,1}, {1,1}, 1, false}, // in=64, out=128, s=2
            Thot::Activation::Raw,
            Thot::Initialization::KaimingNormal
        ),
        Thot::Layer::BatchNorm2d({128, 1e-5, 0.1, true, true}, Thot::Activation::GeLU),
        Thot::Layer::Conv2d(
            {128, 128, {3,3}, {1,1}, {1,1}, {1,1}, 1, false},
            Thot::Activation::Raw,
            Thot::Initialization::KaimingNormal),
        Thot::Layer::BatchNorm2d({128, 1e-5, 0.1, true, true}, Thot::Activation::GeLU),

    }, 1, {
        .projection = Thot::Layer::Conv2d(
            {64, 128, {1,1}, {2,2}, {0,0}, {1,1}, 1, false},   // match stride/channels on skip
            Thot::Activation::Raw,
            Thot::Initialization::KaimingNormal
        )
    }, { .final_activation = Thot::Activation::SiLU }));

    model.add(Thot::Block::Residual({
        Thot::Layer::Conv2d(
            {128, 256, {3,3}, {2,2}, {1,1}, {1,1}, 1, false}, // in=128, out=256, s=2
            Thot::Activation::Raw,
            Thot::Initialization::KaimingNormal
        ),
        Thot::Layer::BatchNorm2d({256, 1e-5, 0.1, true, true}, Thot::Activation::GeLU),
        Thot::Layer::Conv2d(
            {256, 256, {3,3}, {1,1}, {1,1}, {1,1}, 1, false},
            Thot::Activation::Raw,
            Thot::Initialization::KaimingNormal
        )
    }, 1, {
        .projection = Thot::Layer::Conv2d(
            {128, 256, {1,1}, {2,2}, {0,0}, {1,1}, 1, false},
            Thot::Activation::Raw,
            Thot::Initialization::KaimingNormal
        )
    }, { .final_activation = Thot::Activation::SiLU }));

    model.add(Thot::Layer::Dropout({ .probability = 0.3 }));
    model.add(Thot::Layer::AdaptiveAvgPool2d({{1, 1}}));
    model.add(Thot::Layer::Flatten());
    model.add(Thot::Layer::FC({256, 512, true}, Thot::Activation::SiLU, Thot::Initialization::KaimingNormal));
    model.add(Thot::Layer::Dropout({ .probability = 0.5 }));
    model.add(Thot::Layer::FC({512, 10, true}, Thot::Activation::Raw, Thot::Initialization::KaimingNormal));


    const int64_t N = 150000;
    const int64_t B = 128;
    const int64_t epochs = 120;

    const int64_t steps_per_epoch = (N + B - 1) / B;

    model.set_optimizer(
        Thot::Optimizer::AdamW({.learning_rate=1e-3, .weight_decay=5e-4}),
            Thot::LrScheduler::CosineAnnealing({
            .T_max = (epochs-40) * steps_per_epoch,
            .eta_min = 5e-5,
            .warmup_steps = 5*steps_per_epoch,
            .warmup_start_factor = 0.1
        })
    );


    model.set_loss(Thot::Loss::CrossEntropy({.label_smoothing=0.1f}));

    model.set_regularization({Thot::Regularization::SWAG({
        .coefficient = 1e-3,
        .variance_epsilon = 1e-6,
        .start_step = 75*steps_per_epoch,
        .accumulation_stride = 2*steps_per_epoch,
        .max_snapshots = 30,
    })});




    auto [train_images, train_labels, test_images, test_labels] = Thot::Data::Load::CIFAR10("/home/moonfloww/Projects/DATASETS/CIFAR10", 1.f, 1.f, true);
    auto [validation_images, validation_labels] = Thot::Data::Manipulation::Fraction(test_images, test_labels, 0.1f);
    (void)Thot::Data::Check::Size(train_images, "Input train size raw");


    std::tie(train_images, train_labels) = Thot::Data::Manipulation::Cutout(train_images, train_labels, {-1, -1}, {8, 8}, -1, 1.f, true, false);
    std::tie(train_images, train_labels) = Thot::Data::Manipulation::Shuffle(train_images, train_labels);

    std::tie(train_images, train_labels) = Thot::Data::Manipulation::Flip(train_images, train_labels, {"x"}, 0.5f, true, false);
    std::tie(train_images, train_labels) = Thot::Data::Manipulation::Shuffle(train_images, train_labels);

    std::tie(train_images, train_labels) = Thot::Data::Manipulation::Flip(train_images, train_labels, {"y"}, 0.5f, false, false);
    std::tie(train_images, train_labels) = Thot::Data::Manipulation::Shuffle(train_images, train_labels);


    (void)Thot::Data::Check::Size(train_images, "Input train size after augment");


    model.train(train_images, train_labels, {.epoch = epochs, .batch_size = B, .shuffle = true, .buffer_vram = true, .restore_best_state = true, .test = std::make_pair(validation_images, validation_labels)});

    (void) model.evaluate(test_images, test_labels, Thot::Evaluation::Classification,{
        Thot::Metric::Classification::Accuracy,
        Thot::Metric::Classification::Precision,
        Thot::Metric::Classification::Recall,
        Thot::Metric::Classification::F1,
        Thot::Metric::Classification::TruePositiveRate,
        Thot::Metric::Classification::TrueNegativeRate,
        Thot::Metric::Classification::Top1Error,
        Thot::Metric::Classification::ExpectedCalibrationError,
        Thot::Metric::Classification::MaximumCalibrationError,
        Thot::Metric::Classification::CohensKappa,
        Thot::Metric::Classification::LogLoss,
        Thot::Metric::Classification::BrierScore,
    });

    return 0;
}


/*
best test : 48/120
const int64_t N = 150000;
const int64_t B = 128;
const int64_t epochs = 120;
const int64_t warmup_epochs = 5;

const int64_t steps_per_epoch = (N + B - 1) / B;
const int64_t total_steps = epochs * steps_per_epoch;
const int64_t warmup_steps = warmup_epochs * steps_per_epoch;

model.set_optimizer(
Thot::Optimizer::AdamW({.learning_rate=1e-3, .weight_decay=5e-4}),
Thot::LrScheduler::CosineAnnealing({
.T_max = total_steps,
.eta_min = 5e-5,
.warmup_steps = 5*steps_per_epoch,
.warmup_start_factor = 0.1
})
);


model.set_loss(Thot::Loss::CrossEntropy({.label_smoothing=0.1f}));

model.set_regularization({Thot::Regularization::SWAG({
.coefficient = 1e-3,
.variance_epsilon = 1e-6,
.start_step = 75*steps_per_epoch,
.accumulation_stride = 2*steps_per_epoch,
.max_snapshots = 30,
})});
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Evaluation: Classification ┃          ┃                    ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━┫
┃ Metric                     ┃    Macro ┃ Weighted (support) ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━┫
┃ Accuracy                   ┃ 0.864500 ┃           0.864500 ┃
┃ Precision                  ┃ 0.866607 ┃           0.866607 ┃
┃ Recall                     ┃ 0.864500 ┃           0.864500 ┃
┃ F1 score                   ┃ 0.865093 ┃           0.865093 ┃
┃ True positive rate         ┃ 0.864500 ┃           0.864500 ┃
┃ True negative rate         ┃ 0.984944 ┃           0.984944 ┃
┃ Top-1 error                ┃ 0.135500 ┃           0.135500 ┃
┃ Expected calibration error ┃ 0.050568 ┃           0.050568 ┃
┃ Maximum calibration error  ┃ 0.251283 ┃           0.251283 ┃
┃ Cohen's kappa              ┃ 0.849444 ┃           0.849444 ┃
┃ Log loss                   ┃ 0.494737 ┃           0.494737 ┃
┃ Brier score                ┃ 0.200591 ┃           0.200591 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━┛
*/