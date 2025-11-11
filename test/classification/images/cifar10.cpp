#include <iostream>
#include <cstddef>
#include <vector>
#include <torch/torch.h>
#include <utility>
#include "../../../include/Thot.h"

int main() {
    Thot::Model model("Debug_CIFAR");
    model.to_device(torch::cuda::is_available());


    bool IsLoading=false;
    if (IsLoading) {
        model.save("/home/moonfloww/Projects/NNs/CIFAR_DEBUG");
    }
    std::cout << "Cuda: " << torch::cuda::is_available() << std::endl;
    const int64_t N = 200000;
    const int64_t B = std::pow(2,7);
    const int64_t epochs = 20;

    const int64_t steps_per_epoch = (N + B - 1) / B;
    if (!IsLoading) {
        model.add(Thot::Block::Sequential({
            Thot::Layer::Conv2d({3, 64, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false},
                Thot::Activation::Identity, Thot::Initialization::HeNormal
            ),
            Thot::Layer::BatchNorm2d({64, 1e-5, 0.1, true, true},
                Thot::Activation::SiLU
            ),
            Thot::Layer::Conv2d({64, 64, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, true},
                Thot::Activation::Identity, Thot::Initialization::HeNormal,
                {}
            ),
            Thot::Layer::BatchNorm2d({64, 1e-5, 0.1, true, true},
                Thot::Activation::SiLU
            ),
            Thot::Layer::MaxPool2d({{2, 2}, {2, 2}})
        }));

        model.add(Thot::Layer::HardDropout({ .probability = 0.3 }));

        model.add(Thot::Block::Residual({
            Thot::Layer::Conv2d({64, 64, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false},
                Thot::Activation::Identity, Thot::Initialization::HeNormal
            ),
            Thot::Layer::BatchNorm2d({64, 1e-5, 0.1, true, true},
                Thot::Activation::SiLU
            ),
            Thot::Layer::Conv2d({64, 64, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, true},
                Thot::Activation::Identity, Thot::Initialization::HeNormal
            )
        }, 3, {}, { .final_activation = Thot::Activation::SiLU }));

        model.add(Thot::Block::Residual({
            Thot::Layer::Conv2d({64, 128, {3, 3}, {2, 2}, {1, 1}, {1, 1}, 1, false},
                Thot::Activation::Identity, Thot::Initialization::HeNormal
            ),
            Thot::Layer::BatchNorm2d({128, 1e-5, 0.1, true, true},
                Thot::Activation::SiLU
            ),
            Thot::Layer::Conv2d({128, 128, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, true},
                Thot::Activation::Identity, Thot::Initialization::HeNormal
            )
        }, 1,
        {.use_projection = true, .projection = Thot::Layer::Conv2d({64, 128, {1, 1}, {2, 2}, {0, 0}, {1, 1}, 1, false},
                    Thot::Activation::Identity, Thot::Initialization::HeNormal)
            },
            { .final_activation = Thot::Activation::SiLU }));

        model.add(Thot::Layer::HardDropout({ .probability = 0.3 }));

        model.add(Thot::Block::Sequential({
            Thot::Layer::Conv2d({128, 256, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false},
                Thot::Activation::Identity, Thot::Initialization::HeNormal
            ),
            Thot::Layer::BatchNorm2d(
                {256, 1e-5, 0.1, true, true},
                Thot::Activation::SiLU
            ),
            Thot::Layer::AdaptiveAvgPool2d({{1, 1}})
        }));

        model.add(Thot::Block::Sequential({
            Thot::Layer::Conv2d({256, 128, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false},
                Thot::Activation::Identity, Thot::Initialization::HeNormal
            ),
            Thot::Layer::BatchNorm2d({128, 1e-5, 0.1, true, true},
                Thot::Activation::SiLU
            ),
            Thot::Layer::AdaptiveAvgPool2d({{1, 1}})
        }));

        model.add(Thot::Layer::Flatten());

        model.add(Thot::Layer::FC({128, 512, true}, Thot::Activation::SiLU, Thot::Initialization::HeNormal));
        model.add(Thot::Layer::HardDropout({.probability = 0.5}));
        model.add(Thot::Layer::FC({512, 10, true}, Thot::Activation::Identity, Thot::Initialization::HeNormal));


        model.set_optimizer(
            Thot::Optimizer::AdamW({.learning_rate=1e-4, .weight_decay=5e-4}),
                Thot::LrScheduler::CosineAnnealing({
                .T_max = static_cast<size_t>(epochs*0.85) * steps_per_epoch,
                .eta_min = 3e-7,
                .warmup_steps = 5*static_cast<size_t>(steps_per_epoch),
                .warmup_start_factor = 0.1
            })
        );

        model.set_loss(Thot::Loss::CrossEntropy({.label_smoothing=0.02f}));

        model.set_regularization({ Thot::Regularization::SWAG({
            .coefficient = 1e-3,
            .variance_epsilon = 1e-6,
            .start_step = static_cast<size_t>(0.85 * (steps_per_epoch*epochs)),
            .accumulation_stride = static_cast<size_t>(steps_per_epoch),
            .max_snapshots = 20,
        })});


    }

    auto [train_images, train_labels, test_images, test_labels] = Thot::Data::Load::CIFAR10("/home/moonfloww/Projects/DATASETS/CIFAR10", 1.f, 1.f, true);
    auto [validation_images, validation_labels] = Thot::Data::Manipulation::Fraction(test_images, test_labels, 0.1f);
    Thot::Data::Check::Size(train_images, "Raw");
    (void)Thot::Plot::Data::Image(train_images, {1,2,3,4,5,6,7});

    std::tie(train_images, train_labels) = Thot::Data::Manipulation::Cutout(train_images, train_labels, {-1, -1}, {12, 12}, -1, 1.f, true, false);
    std::tie(train_images, train_labels) = Thot::Data::Manipulation::Shuffle(train_images, train_labels);

    std::tie(train_images, train_labels) = Thot::Data::Manipulation::Flip(train_images, train_labels, {"x"}, 1.f, true, false);
    std::tie(train_images, train_labels) = Thot::Data::Manipulation::Shuffle(train_images, train_labels);

    Thot::Data::Check::Size(train_images, "Augmented");


    if (!IsLoading) {
        model.train(train_images, train_labels, {
            .epoch = static_cast<std::size_t>(epochs),
            .batch_size = static_cast<std::size_t>(B),
            .shuffle = true,
            .restore_best_state = true,
            .validation = std::vector<at::Tensor>{validation_images, validation_labels},
            .graph_mode = Thot::GraphMode::Capture,
            .enable_amp=true,
            .memory_format = torch::MemoryFormat::Contiguous}
        );
    }

    model.evaluate(test_images, test_labels, Thot::Evaluation::Classification,{
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
    }, {.batch_size = 64});



    return 0;
}


/*
Epoch [1/20] | Train loss: 1.969582 | Test loss: 1.555169 | ΔLoss: N/A (∇) | duration: 11.05sec
Epoch [2/20] | Train loss: 1.506606 | Test loss: 1.313850 | ΔLoss: -0.241319 (∇) | duration: 10.74sec
Epoch [3/20] | Train loss: 1.319500 | Test loss: 1.192241 | ΔLoss: -0.121609 (∇) | duration: 10.63sec
Epoch [4/20] | Train loss: 1.167014 | Test loss: 1.053578 | ΔLoss: -0.138663 (∇) | duration: 10.04sec
Epoch [5/20] | Train loss: 1.038362 | Test loss: 0.941287 | ΔLoss: -0.112291 (∇) | duration: 10.03sec
Epoch [6/20] | Train loss: 0.942045 | Test loss: 0.868643 | ΔLoss: -0.072644 (∇) | duration: 10.02sec
Epoch [7/20] | Train loss: 0.867824 | Test loss: 0.828729 | ΔLoss: -0.039914 (∇) | duration: 10.05sec
Epoch [8/20] | Train loss: 0.810952 | Test loss: 0.822771 | ΔLoss: -0.005958 (∇) | duration: 10.04sec
Epoch [9/20] | Train loss: 0.762011 | Test loss: 0.795204 | ΔLoss: -0.027567 (∇) | duration: 10.03sec
Epoch [10/20] | Train loss: 0.724816 | Test loss: 0.766069 | ΔLoss: -0.029134 (∇) | duration: 10.02sec
Epoch [11/20] | Train loss: 0.690788 | Test loss: 0.741530 | ΔLoss: -0.024540 (∇) | duration: 10.04sec
Epoch [12/20] | Train loss: 0.659150 | Test loss: 0.732725 | ΔLoss: -0.008805 (∇) | duration: 10.04sec
Epoch [13/20] | Train loss: 0.634025 | Test loss: 0.709602 | ΔLoss: -0.023123 (∇) | duration: 10.06sec
Epoch [14/20] | Train loss: 0.610396 | Test loss: 0.702317 | ΔLoss: -0.007285 (∇) | duration: 10.03sec
Epoch [15/20] | Train loss: 0.590512 | Test loss: 0.695988 | ΔLoss: -0.006329 (∇) | duration: 10.03sec
Epoch [16/20] | Train loss: 0.573686 | Test loss: 0.696038 | ΔLoss: +0.000051 (∇) | duration: 10.03sec
Epoch [17/20] | Train loss: 0.560695 | Test loss: 0.697579 | ΔLoss: +0.001592 (∇) | duration: 10.02sec
Epoch [18/20] | Train loss: 0.547826 | Test loss: 0.697189 | ΔLoss: +0.001202 (∇) | duration: 10.02sec
Epoch [19/20] | Train loss: 0.541938 | Test loss: 0.694397 | ΔLoss: -0.001591 (∇) | duration: 12.86sec
Epoch [20/20] | Train loss: 0.534568 | Test loss: 0.692383 | ΔLoss: -0.002013 (∇) | duration: 12.84sec




Epoch [1/80] | Train loss: 2.043670 | Test loss: 1.558099 | ΔLoss: N/A (∇) | duration: 21.43sec
Epoch [2/80] | Train loss: 1.551665 | Test loss: 1.325052 | ΔLoss: -0.233048 (∇) | duration: 22.25sec
Epoch [3/80] | Train loss: 1.316544 | Test loss: 1.125911 | ΔLoss: -0.199141 (∇) | duration: 20.96sec
Epoch [4/80] | Train loss: 1.153605 | Test loss: 1.000342 | ΔLoss: -0.125569 (∇) | duration: 22.22sec
Epoch [5/80] | Train loss: 1.035271 | Test loss: 0.889793 | ΔLoss: -0.110549 (∇) | duration: 23.31sec
Epoch [6/80] | Train loss: 0.938822 | Test loss: 0.797055 | ΔLoss: -0.092738 (∇) | duration: 23.13sec
Epoch [7/80] | Train loss: 0.856046 | Test loss: 0.760538 | ΔLoss: -0.036517 (∇) | duration: 22.97sec
Epoch [8/80] | Train loss: 0.791008 | Test loss: 0.670695 | ΔLoss: -0.089843 (∇) | duration: 21.81sec
Epoch [9/80] | Train loss: 0.738124 | Test loss: 0.638725 | ΔLoss: -0.031971 (∇) | duration: 24.40sec
Epoch [10/80] | Train loss: 0.695576 | Test loss: 0.600487 | ΔLoss: -0.038238 (∇) | duration: 23.00sec
Epoch [11/80] | Train loss: 0.659483 | Test loss: 0.595433 | ΔLoss: -0.005054 (∇) | duration: 22.78sec
Epoch [12/80] | Train loss: 0.626604 | Test loss: 0.572415 | ΔLoss: -0.023018 (∇) | duration: 24.25sec
Epoch [13/80] | Train loss: 0.599530 | Test loss: 0.567682 | ΔLoss: -0.004733 (∇) | duration: 22.31sec
Epoch [14/80] | Train loss: 0.572129 | Test loss: 0.545594 | ΔLoss: -0.022088 (∇) | duration: 21.55sec
 --------------------------------------CIFAR10--------------------------------------
        model.add(Thot::Block::Sequential({
            Thot::Layer::Conv2d(
                {3, 64, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false},
                Thot::Activation::Identity,
                Thot::Initialization::HeNormal
            ),
            Thot::Layer::BatchNorm2d(
                {64, 1e-5, 0.1, true, true},
                Thot::Activation::SiLU
            ),
            Thot::Layer::Conv2d(
                {64, 64, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, true},
                Thot::Activation::Identity,
                Thot::Initialization::HeNormal,
                {}
            ),
            Thot::Layer::BatchNorm2d(
                {64, 1e-5, 0.1, true, true},
                Thot::Activation::SiLU
            ),
            Thot::Layer::MaxPool2d({{2, 2}, {2, 2}})
        }));

        model.add(Thot::Layer::HardDropout({ .probability = 0.3 }));


        model.add(Thot::Block::Residual({
            Thot::Layer::Conv2d(
                {64, 64, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false},
                Thot::Activation::Identity,
                Thot::Initialization::HeNormal
            ),
            Thot::Layer::BatchNorm2d(
                {64, 1e-5, 0.1, true, true},
                Thot::Activation::SiLU
            ),
            Thot::Layer::Conv2d(
                {64, 64, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, true},
                Thot::Activation::Identity,
                Thot::Initialization::HeNormal
            )
        }, 3, {}, { .final_activation = Thot::Activation::SiLU }));

        model.add(Thot::Block::Residual({
            Thot::Layer::Conv2d(
                {64, 128, {3, 3}, {2, 2}, {1, 1}, {1, 1}, 1, false},
                Thot::Activation::Identity,
                Thot::Initialization::HeNormal
            ),
            Thot::Layer::BatchNorm2d(
                {128, 1e-5, 0.1, true, true},
                Thot::Activation::SiLU
            ),
            Thot::Layer::Conv2d(
                {128, 128, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, true},
                Thot::Activation::Identity,
                Thot::Initialization::HeNormal
            )
        }, 1,
        {.use_projection = true, .projection = Thot::Layer::Conv2d(
                    {64, 128, {1, 1}, {2, 2}, {0, 0}, {1, 1}, 1, false},
                    Thot::Activation::Identity,
                    Thot::Initialization::HeNormal)
            },
            { .final_activation = Thot::Activation::SiLU }));

        model.add(Thot::Layer::HardDropout({ .probability = 0.3 }));

        model.add(Thot::Block::Sequential({
            Thot::Layer::Conv2d(
                {128, 256, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false},
                Thot::Activation::Identity,
                Thot::Initialization::HeNormal
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
                Thot::Activation::Identity,
                Thot::Initialization::HeNormal
            ),
            Thot::Layer::BatchNorm2d(
                {128, 1e-5, 0.1, true, true},
                Thot::Activation::SiLU
            ),
            Thot::Layer::AdaptiveAvgPool2d({{1, 1}})
        }));

        model.add(Thot::Layer::Flatten());

        model.add(Thot::Layer::FC({128, 512, true}, Thot::Activation::SiLU, Thot::Initialization::HeNormal));
        model.add(Thot::Layer::HardDropout({.probability = 0.5}));
        model.add(Thot::Layer::FC({512, 10, true}, Thot::Activation::Identity, Thot::Initialization::HeNormal));


        model.set_optimizer(
            Thot::Optimizer::AdamW({.learning_rate=1e-4, .weight_decay=5e-4}),
                Thot::LrScheduler::CosineAnnealing({
                .T_max = static_cast<size_t>(epochs*0.85) * steps_per_epoch,
                .eta_min = 3e-7,
                .warmup_steps = 5*static_cast<size_t>(steps_per_epoch),
                .warmup_start_factor = 0.1
            })
        );

        model.set_loss(Thot::Loss::CrossEntropy({.label_smoothing=0.02f}));

        model.set_regularization({ Thot::Regularization::SWAG({
          .coefficient = 1e-3,
          .variance_epsilon = 1e-6,
          .start_step = static_cast<size_t>(0.85 * (steps_per_epoch*epochs)),
          .accumulation_stride = static_cast<size_t>(steps_per_epoch),
          .max_snapshots = 20,
        })});


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Evaluation: Classification ┃          ┃                    ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━┫
┃ Metric                     ┃    Macro ┃ Weighted (support) ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━┫
┃ Accuracy                   ┃ 0.974560 ┃           0.974560 ┃
┃ Precision                  ┃ 0.872237 ┃           0.872237 ┃
┃ Recall                     ┃ 0.872800 ┃           0.872800 ┃
┃ F1 score                   ┃ 0.872213 ┃           0.872213 ┃
┃ True positive rate         ┃ 0.872800 ┃           0.872800 ┃
┃ True negative rate         ┃ 0.985867 ┃           0.985867 ┃
┃ Top-1 error                ┃ 0.127200 ┃           0.127200 ┃
┃ Expected calibration error ┃ 0.037887 ┃           0.037887 ┃
┃ Maximum calibration error  ┃ 0.148656 ┃           0.148656 ┃
┃ Cohen's kappa              ┃ 0.858667 ┃           0.858667 ┃
┃ Log loss                   ┃ 0.415918 ┃           0.415918 ┃
┃ Brier score                ┃ 0.190100 ┃           0.190100 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━┛





stresstest:
    model.add(Thot::Layer::Conv2d({.in_channels = 3,  .out_channels = 32,  .kernel_size = {3,3}, .stride={1,1}, .padding={1,1}, .dilation={1,1}}, Thot::Activation::ReLU, Thot::Initialization::HeNormal));
    model.add(Thot::Layer::Conv2d({.in_channels = 32, .out_channels = 32,  .kernel_size = {3,3}, .stride={1,1}, .padding={1,1}, .dilation={1,1}}, Thot::Activation::ReLU, Thot::Initialization::HeNormal));
    model.add(Thot::Layer::MaxPool2d({.kernel_size={2,2}, .stride={2,2}, .padding={0,0}}));

    model.add(Thot::Layer::Conv2d({.in_channels = 32, .out_channels = 64,  .kernel_size = {3,3}, .stride={1,1}, .padding={1,1}, .dilation={1,1}}, Thot::Activation::ReLU, Thot::Initialization::HeNormal));
    model.add(Thot::Layer::Conv2d({.in_channels = 64, .out_channels = 64,  .kernel_size = {3,3}, .stride={1,1}, .padding={1,1}, .dilation={1,1}}, Thot::Activation::ReLU, Thot::Initialization::HeNormal));
    model.add(Thot::Layer::MaxPool2d({.kernel_size={2,2}, .stride={2,2}, .padding={0,0}}));

    model.add(Thot::Layer::Conv2d({.in_channels = 64, .out_channels = 128, .kernel_size = {3,3}, .stride={1,1}, .padding={1,1}, .dilation={1,1}}, Thot::Activation::ReLU, Thot::Initialization::HeNormal));
    model.add(Thot::Layer::MaxPool2d({.kernel_size={2,2}, .stride={2,2}, .padding={0,0}}));

    model.add(Thot::Layer::Flatten());
    model.add(Thot::Layer::HardDropout({0.1f}));
    model.add(Thot::Layer::FC({2048, 524}, Thot::Activation::ReLU,  Thot::Initialization::HeNormal));
    model.add(Thot::Layer::FC({524, 126},  Thot::Activation::ReLU,  Thot::Initialization::HeNormal));
    model.add(Thot::Layer::FC({126, 10},   Thot::Activation::Identity, Thot::Initialization::HeNormal));


    model.set_loss(Thot::Loss::CrossEntropy());
    model.set_optimizer(Thot::Optimizer::SGD({3e-5f}));

    auto [train_images, train_labels, test_images, test_labels] = Thot::Data::Load::CIFAR10("/home/moonfloww/Projects/DATASETS/CIFAR10", .05f, 1.f, true);

    model.train(train_images, train_labels, {.epoch = 10, .batch_size = 64});

*/