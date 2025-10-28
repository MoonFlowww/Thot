#include <iostream>
#include <cstddef>
#include <vector>
#include <torch/torch.h>
#include <utility>
#include "../include/Thot.h"

int main() {
    Thot::Model model("Debug_CIFAR");
    bool IsLoading=false;
    if (IsLoading) {
        model.save("/home/moonfloww/Projects/NNs/CIFAR_DEBUG");
    }
    std::cout << "Cuda: " << torch::cuda::is_available() << std::endl;
    model.to_device(torch::cuda::is_available());
    const int64_t N = 200000;
    const int64_t B = std::pow(2,7);
    const int64_t epochs = 80;

    const int64_t steps_per_epoch = (N + B - 1) / B;
    if (!IsLoading) {
        model.add(Thot::Block::Sequential({
            Thot::Layer::Conv2d(
                {3, 64, {3,3}, {1,1}, {1,1}, {1,1}, 1, false},
                Thot::Activation::Identity,
                Thot::Initialization::KaimingNormal
            ),
            Thot::Layer::BatchNorm2d({64, 1e-5, 0.1, true, true}, Thot::Activation::GeLU),
            Thot::Layer::Conv2d(
                {64, 64, {3,3}, {1,1}, {1,1}, {1,1}, 1, false},
                Thot::Activation::Identity,
                Thot::Initialization::KaimingNormal
            ),
            Thot::Layer::BatchNorm2d({64, 1e-5, 0.1, true, true}, Thot::Activation::GeLU),
            Thot::Layer::MaxPool2d({{2,2}, {2,2}})
        }));

        model.add(Thot::Layer::SoftDropout({ .probability = 0.08, .noise_mean = 1.0, .noise_std = 0.03 }));

        model.add(Thot::Block::Residual({
            Thot::Layer::BatchNorm2d({64, 1e-5, 0.1, true, true}, Thot::Activation::GeLU),
            Thot::Layer::Conv2d({64,128,{3,3},{2,2},{1,1},{1,1},1,false}, Thot::Activation::Identity, Thot::Initialization::KaimingNormal),
            Thot::Layer::BatchNorm2d({128,1e-5,0.1,true,true}, Thot::Activation::GeLU),
            Thot::Layer::Conv2d({128,128,{3,3},{1,1},{1,1},{1,1},1,false},Thot::Activation::Identity,Thot::Initialization::KaimingNormal)
        }, 1, {
            .projection = Thot::Layer::Conv2d({64,128,{1,1},{2,2},{0,0},{1,1},1,false}, Thot::Activation::Identity, Thot::Initialization::KaimingNormal)
        }, { .final_activation = Thot::Activation::Identity }));

        model.add(Thot::Layer::SoftDropout({ .probability = 0.08, .noise_mean = 1.0, .noise_std = 0.03 }));

        model.add(Thot::Block::Residual({
            Thot::Layer::BatchNorm2d({128,1e-5,0.1,true,true}, Thot::Activation::GeLU),
            Thot::Layer::Conv2d({128,128,{3,3},{1,1},{1,1},{1,1},1,false},Thot::Activation::Identity ,Thot::Initialization::KaimingNormal),
            Thot::Layer::BatchNorm2d({128,1e-5,0.1,true,true}, Thot::Activation::GeLU),
            Thot::Layer::Conv2d({128,128,{3,3},{1,1},{1,1},{1,1},1,false}, Thot::Activation::Identity, Thot::Initialization::KaimingNormal)
        }, 1, {}, { .final_activation = Thot::Activation::Identity }));

        model.add(Thot::Block::Residual({
            Thot::Layer::BatchNorm2d({128,1e-5,0.1,true,true}, Thot::Activation::GeLU),
            Thot::Layer::Conv2d({128,128,{3,3},{1,1},{1,1},{1,1},1,false}, Thot::Activation::Identity, Thot::Initialization::KaimingNormal),
            Thot::Layer::BatchNorm2d({128,1e-5,0.1,true,true}, Thot::Activation::GeLU),
            Thot::Layer::Conv2d({128,128,{3,3},{1,1},{1,1},{1,1},1,false}, Thot::Activation::Identity, Thot::Initialization::KaimingNormal)
        }, 1, {}, { .final_activation = Thot::Activation::Identity }));

        model.add(Thot::Layer::SoftDropout({ .probability = 0.1, .noise_mean = 1.0, .noise_std = 0.08 }));

        model.add(Thot::Block::Residual({
            Thot::Layer::BatchNorm2d({128,1e-5,0.1,true,true}, Thot::Activation::GeLU),
            Thot::Layer::Conv2d({128,256,{3,3},{2,2},{1,1},{1,1},1,false}, Thot::Activation::Identity, Thot::Initialization::KaimingNormal),
            Thot::Layer::BatchNorm2d({256,1e-5,0.1,true,true}, Thot::Activation::GeLU),
            Thot::Layer::Conv2d({256,256,{3,3},{1,1},{1,1},{1,1},1,false}, Thot::Activation::Identity, Thot::Initialization::KaimingNormal)
        }, 1, {
            .projection = Thot::Layer::Conv2d({128,256,{1,1},{2,2},{0,0},{1,1},1,false}, Thot::Activation::Identity, Thot::Initialization::KaimingNormal)
        }, { .final_activation = Thot::Activation::Identity }));

        model.add(Thot::Layer::SoftDropout({ .probability = 0.1, .noise_mean = 1.0, .noise_std = 0.08 }));

        model.add(Thot::Block::Residual({
            Thot::Layer::BatchNorm2d({256,1e-5,0.1,true,true}, Thot::Activation::GeLU),
            Thot::Layer::Conv2d({256,256,{3,3},{1,1},{1,1},{1,1},1,false}, Thot::Activation::Identity, Thot::Initialization::KaimingNormal),
            Thot::Layer::BatchNorm2d({256,1e-5,0.1,true,true}, Thot::Activation::GeLU),
            Thot::Layer::Conv2d({256,256,{3,3},{1,1},{1,1},{1,1},1,false}, Thot::Activation::Identity, Thot::Initialization::KaimingNormal)
        }, 1, {}, { .final_activation = Thot::Activation::Identity }));


        model.add(Thot::Layer::SoftDropout({ .probability = 0.1, .noise_mean = 1.0, .noise_std = 0.12 }));
        model.add(Thot::Layer::AdaptiveAvgPool2d({{1, 1}}));
        model.add(Thot::Layer::Flatten());
        model.add(Thot::Layer::FC({256, 10, true}, Thot::Activation::SiLU, Thot::Initialization::KaimingNormal));
        model.add(Thot::Layer::HardDropout({ .probability = 0.33 }));
        model.add(Thot::Layer::FC({10, 10, true}, Thot::Activation::Identity, Thot::Initialization::KaimingNormal));




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
    Thot::Data::Check::Size(train_images, "Input train size raw");


    std::tie(train_images, train_labels) = Thot::Data::Manipulation::Cutout(train_images, train_labels, {-1, -1}, {12, 12}, -1, 1.f, true, false);
    std::tie(train_images, train_labels) = Thot::Data::Manipulation::Shuffle(train_images, train_labels);

    std::tie(train_images, train_labels) = Thot::Data::Manipulation::Flip(train_images, train_labels, {"x"}, 1.f, true, false);
    std::tie(train_images, train_labels) = Thot::Data::Manipulation::Shuffle(train_images, train_labels);

    Thot::Data::Check::Size(train_images, "Input train size after augment");

    if (!IsLoading) {
        model.train(train_images, train_labels, {.epoch = epochs, .batch_size = B, .shuffle = true, .buffer_vram = 0, .restore_best_state = true,
                        .test = std::make_pair(validation_images, validation_labels)});
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
    });


    if (!IsLoading) {
        model.calibrate(train_images, train_labels, {Thot::Calibration::TemperatureScalingDescriptor{}}, true, std::make_pair(train_images, train_labels)); // std::make_pair(test_images, test_labels)
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
    });


    try {
        model.plot(Thot::Plot::Reliability::ROC({
                        .KSTest = true,
                        .thresholds = true,
                        .adjustScale = false,
                    }),
                    train_images,
                    train_labels,
                    test_images,
                    test_labels);

        model.plot(Thot::Plot::Reliability::PR({
                        .samples = true,
                        .random = false,
                        .interpolate = true,
                        .adjustScale = false,
                    }),
                    train_images,
                    train_labels,
                    test_images,
                    test_labels);

        model.plot(Thot::Plot::Reliability::DET({
                        .KSTest = true,
                        .confidenceBands = true,
                        .annotateCrossing = true,
                        .adjustScale = false,
                    }),
                    train_images,
                    train_labels,
                    test_images,
                    test_labels);
    } catch (const std::exception& plotting_error) {
        std::cerr << "Plotting failed: " << plotting_error.what() << std::endl;
    }
    model.save("/home/moonfloww/Projects/NNs/CIFAR_DEBUG");
    return 0;
}


/*

model.set_optimizer(
    Thot::Optimizer::AdamW({.learning_rate=1e-3, .weight_decay=5e-4}),
        Thot::LrScheduler::CosineAnnealing({
        .T_max = (epochs) * steps_per_epoch,
        .eta_min = 3e-7,
        .warmup_steps = 5*steps_per_epoch,
        .warmup_start_factor = 0.1
    })
);


model.set_loss(Thot::Loss::CrossEntropy({.label_smoothing=0.05f}));

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
┃ Accuracy                   ┃ 0.882600 ┃           0.882600 ┃
┃ Precision                  ┃ 0.882892 ┃           0.882892 ┃
┃ Recall                     ┃ 0.882600 ┃           0.882600 ┃
┃ F1 score                   ┃ 0.881857 ┃           0.881857 ┃
┃ True positive rate         ┃ 0.882600 ┃           0.882600 ┃
┃ True negative rate         ┃ 0.986956 ┃           0.986956 ┃
┃ Top-1 error                ┃ 0.117400 ┃           0.117400 ┃
┃ Expected calibration error ┃ 0.034671 ┃           0.034671 ┃
┃ Maximum calibration error  ┃ 0.244972 ┃           0.244972 ┃
┃ Cohen's kappa              ┃ 0.869556 ┃           0.869556 ┃
┃ Log loss                   ┃ 0.451599 ┃           0.451599 ┃
┃ Brier score                ┃ 0.185568 ┃           0.185568 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━┛

*/