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
 *Epoch [19/20] | Train loss: 0.445165 | Test loss: 0.542593 | ΔLoss: -0.001442 (∇) | duration: 23.38sec
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Evaluation: Classification ┃          ┃                    ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━┫
┃ Metric                     ┃  Macro   ┃ Weighted (support) ┃
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
*/