#include <iostream>
#include <cstddef>
#include <vector>
#include <torch/torch.h>
#include <utility>
#include "../include/Thot.h"




int main() {
    Thot::Model model;
    model.to_device(torch::cuda::is_available());

    model.add(Thot::Block::Sequential({
        Thot::Layer::Conv2d(
            {3, 64, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false},
            Thot::Activation::Raw,
            Thot::Initialization::KaimingNormal
        ),
        Thot::Layer::BatchNorm2d(
            {64, 1e-5, 0.1, true, true},
            Thot::Activation::SiLU
        ),
        Thot::Layer::Conv2d(
            {64, 64, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, true},
            Thot::Activation::Raw,
            Thot::Initialization::KaimingNormal
        ),
        Thot::Layer::BatchNorm2d(
            {64, 1e-5, 0.1, true, true},
            Thot::Activation::SiLU
        ),
        Thot::Layer::MaxPool2d({{2, 2}, {2, 2}})
    }));

    model.add(Thot::Layer::Dropout({ .probability = 0.3 }));

    model.add(Thot::Block::Residual({
        Thot::Layer::Conv2d(
            {64, 128, {3, 3}, {2, 2}, {1, 1}, {1, 1}, 1, false},
            Thot::Activation::Raw,
            Thot::Initialization::KaimingNormal
        ),
        Thot::Layer::BatchNorm2d(
            {128, 1e-5, 0.1, true, true},
            Thot::Activation::SiLU
        ),
        Thot::Layer::Conv2d(
            {128, 128, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, true},
            Thot::Activation::Raw,
            Thot::Initialization::KaimingNormal
        )
        }));

    model.add(Thot::Block::Residual({
        Thot::Layer::Conv2d(
        {128, 128, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false},
            Thot::Activation::Raw,
            Thot::Initialization::KaimingNormal
        ),
        Thot::Layer::BatchNorm2d(
            {128, 1e-5, 0.1, true, true},
            Thot::Activation::SiLU
        ),
        Thot::Layer::Conv2d(
            {128, 128, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, true},
            Thot::Activation::Raw,
            Thot::Initialization::KaimingNormal
        )
    }, 1, {}, { .final_activation = Thot::Activation::SiLU }));

    model.add(Thot::Layer::Dropout({ .probability = 0.3 }));

    model.add(Thot::Block::Sequential({
        Thot::Layer::Conv2d(
            {128, 256, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false},
            Thot::Activation::Raw,
            Thot::Initialization::KaimingNormal
        ),
        Thot::Layer::BatchNorm2d(
            {256, 1e-5, 0.1, true, true},
            Thot::Activation::SiLU
        ),
        Thot::Layer::AdaptiveAvgPool2d({{1, 1}})
    }));

    model.add(Thot::Block::Sequential({
        Thot::Layer::Conv2d(
            {256, 128, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false},
            Thot::Activation::Raw,
            Thot::Initialization::KaimingNormal
        ),
        Thot::Layer::BatchNorm2d(
            {128, 1e-5, 0.1, true, true},
            Thot::Activation::SiLU
        ),
        Thot::Layer::AdaptiveAvgPool2d({{1, 1}})
    }));

    model.add(Thot::Layer::Flatten());

    model.add(Thot::Layer::FC({128, 512, true}, Thot::Activation::SiLU, Thot::Initialization::KaimingNormal));
    model.add(Thot::Layer::Dropout({.probability = 0.5}));
    model.add(Thot::Layer::FC({512, 10, true}, Thot::Activation::Raw, Thot::Initialization::KaimingNormal));



    model.set_optimizer(
    Thot::Optimizer::AdamW({.learning_rate = 1e-3, .weight_decay = 2e-2}),
    Thot::LrScheduler::CosineAnnealing({ .T_max = 200 * ((50000+256-1)/256), .eta_min = 5e-6, .warmup_steps = 5 * ((50000+256-1)/256), .warmup_start_factor = 0.1}));
    model.set_loss(Thot::Loss::CrossEntropy({.label_smoothing=0.1f}));

    model.set_regularization({Thot::Regularization::SWAG({
            .coefficient = 1e-3,
            .variance_epsilon = 1e-6,
            .start_step = 55,
            .accumulation_stride = 2,
            .max_snapshots = 30,
        })});




    auto [train_images, train_labels, test_images, test_labels] = Thot::Data::Load::CIFAR10("/home/moonfloww/Projects/DATASETS/CIFAR10", 1.f, 1.f, true);
    auto [validation_images, validation_labels] = Thot::Data::Manipulation::Fraction(test_images, test_labels, 0.1f);
    (void)Thot::Data::Check::Size(train_images, "Input train size raw");


    std::tie(train_images, train_labels) = Thot::Data::Manipulation::Cutout(train_images, train_labels, {-1, -1}, {8, 8}, -1, 1.f, true, false);
    std::tie(train_images, train_labels) = Thot::Data::Manipulation::Flip(train_images, train_labels, {"x"}, 0.5f, true, false);
    std::tie(train_images, train_labels) = Thot::Data::Manipulation::Flip(train_images, train_labels, {"y"}, 0.5f, false, false);

    (void)Thot::Data::Check::Size(train_images, "Input train size after augment");

    (void) model.check();

    model.train(train_images, train_labels, {.epoch = 200, .batch_size = 128, .shuffle = false, .restore_best_state = true, .test = std::make_pair(validation_images, validation_labels)});

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
