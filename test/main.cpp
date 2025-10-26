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
        model.save("/home/moonfloww/Projects/NNs");
    }
    std::cout << "Cuda: " << torch::cuda::is_available() << std::endl;
    model.to_device(torch::cuda::is_available());
    const int64_t N = 200000;
    const int64_t B = 258;
    const int64_t epochs = 15;

    const int64_t steps_per_epoch = (N + B - 1) / B;
    if (!IsLoading) {
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
            Thot::Layer::BatchNorm2d({64, 1e-5, 0.1, true, true}, Thot::Activation::GeLU),
            Thot::Layer::Conv2d({64,128,{3,3},{2,2},{1,1},{1,1},1,false}, Thot::Activation::Raw, Thot::Initialization::KaimingNormal),
            Thot::Layer::BatchNorm2d({128,1e-5,0.1,true,true}, Thot::Activation::GeLU),
            Thot::Layer::Conv2d({128,128,{3,3},{1,1},{1,1},{1,1},1,false},Thot::Activation::Raw,Thot::Initialization::KaimingNormal)
        }, 1, {
            .projection = Thot::Layer::Conv2d({64,128,{1,1},{2,2},{0,0},{1,1},1,false}, Thot::Activation::Raw, Thot::Initialization::KaimingNormal)
        }, { .final_activation = Thot::Activation::Identity }));



        model.add(Thot::Block::Residual({
            Thot::Layer::BatchNorm2d({128,1e-5,0.1,true,true}, Thot::Activation::GeLU),
            Thot::Layer::Conv2d({128,128,{3,3},{1,1},{1,1},{1,1},1,false},Thot::Activation::Raw,Thot::Initialization::KaimingNormal),
            Thot::Layer::BatchNorm2d({128,1e-5,0.1,true,true}, Thot::Activation::GeLU),
            Thot::Layer::Conv2d({128,128,{3,3},{1,1},{1,1},{1,1},1,false},Thot::Activation::Raw,Thot::Initialization::KaimingNormal)
        }, 1, {}, { .final_activation = Thot::Activation::Identity }));

        model.add(Thot::Block::Residual({
            Thot::Layer::BatchNorm2d({128,1e-5,0.1,true,true}, Thot::Activation::GeLU),
            Thot::Layer::Conv2d({128,128,{3,3},{1,1},{1,1},{1,1},1,false},Thot::Activation::Raw,Thot::Initialization::KaimingNormal),
            Thot::Layer::BatchNorm2d({128,1e-5,0.1,true,true}, Thot::Activation::GeLU),
            Thot::Layer::Conv2d({128,128,{3,3},{1,1},{1,1},{1,1},1,false},Thot::Activation::Raw,Thot::Initialization::KaimingNormal)
        }, 1, {}, { .final_activation = Thot::Activation::Identity }));



        // --- Stage 3: 128 -> 256 (downsample), then two identity blocks ---
        model.add(Thot::Block::Residual({
            Thot::Layer::BatchNorm2d({128,1e-5,0.1,true,true}, Thot::Activation::GeLU),
            Thot::Layer::Conv2d({128,256,{3,3},{2,2},{1,1},{1,1},1,false},Thot::Activation::Raw,Thot::Initialization::KaimingNormal),
            Thot::Layer::BatchNorm2d({256,1e-5,0.1,true,true}, Thot::Activation::GeLU),
            Thot::Layer::Conv2d({256,256,{3,3},{1,1},{1,1},{1,1},1,false},Thot::Activation::Raw,Thot::Initialization::KaimingNormal)
        }, 1, {
            .projection = Thot::Layer::Conv2d({128,256,{1,1},{2,2},{0,0},{1,1},1,false}, Thot::Activation::Raw, Thot::Initialization::KaimingNormal)
        }, { .final_activation = Thot::Activation::Identity }));

        for (int i=0;i<2;++i) {
            model.add(Thot::Block::Residual({
                Thot::Layer::BatchNorm2d({256,1e-5,0.1,true,true}, Thot::Activation::GeLU),
                Thot::Layer::Conv2d({256,256,{3,3},{1,1},{1,1},{1,1},1,false},Thot::Activation::Raw,Thot::Initialization::KaimingNormal),
                Thot::Layer::BatchNorm2d({256,1e-5,0.1,true,true}, Thot::Activation::GeLU),
                Thot::Layer::Conv2d({256,256,{3,3},{1,1},{1,1},{1,1},1,false},Thot::Activation::Raw,Thot::Initialization::KaimingNormal)
            }, 1, {/* no projection */}, { .final_activation = Thot::Activation::Identity }));
        }

        model.add(Thot::Layer::Dropout({ .probability = 0.3 }));
        model.add(Thot::Layer::AdaptiveAvgPool2d({{1, 1}}));
        model.add(Thot::Layer::Flatten());
        model.add(Thot::Layer::FC({256, 512, true}, Thot::Activation::SiLU, Thot::Initialization::KaimingNormal));
        model.add(Thot::Layer::Dropout({ .probability = 0.5 }));
        model.add(Thot::Layer::FC({512, 10, true}, Thot::Activation::Raw, Thot::Initialization::KaimingNormal));




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
    }

    auto [train_images, train_labels, test_images, test_labels] = Thot::Data::Load::CIFAR10("/home/moonfloww/Projects/DATASETS/CIFAR10", 1.f, 1.f, true);
    auto [validation_images, validation_labels] = Thot::Data::Manipulation::Fraction(test_images, test_labels, 0.1f);
    (void)Thot::Data::Check::Size(train_images, "Input train size raw");


    std::tie(train_images, train_labels) = Thot::Data::Manipulation::Cutout(train_images, train_labels, {-1, -1}, {8, 8}, -1, 1.f, true, false);
    std::tie(train_images, train_labels) = Thot::Data::Manipulation::Shuffle(train_images, train_labels);

    std::tie(train_images, train_labels) = Thot::Data::Manipulation::Flip(train_images, train_labels, {"x"}, 1.f, true, false);
    std::tie(train_images, train_labels) = Thot::Data::Manipulation::Shuffle(train_images, train_labels);

    (void)Thot::Data::Check::Size(train_images, "Input train size after augment");

    if (!IsLoading) {
        model.train(train_images, train_labels, {
                        .epoch = epochs, .batch_size = B, .shuffle = true, .buffer_vram = 0, .restore_best_state = true,
                        .test = std::make_pair(validation_images, validation_labels)
                    });
    }

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


    if (!IsLoading) {
        model.calibrate(train_images, train_labels, {Thot::Calibration::TemperatureScalingDescriptor{}}, true, std::make_pair(test_images, test_labels));
    }


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

    model.save("/home/moonfloww/Projects/NNs");
    return 0;
}


/*

    const int64_t N = 200000;
    const int64_t B = 258;
    const int64_t epochs = 15;

    const int64_t steps_per_epoch = (N + B - 1) / B;

    model.set_optimizer(
        Thot::Optimizer::AdamW({.learning_rate=1e-3, .weight_decay=5e-4}),
            Thot::LrScheduler::CosineAnnealing({
            .T_max = (epochs) * steps_per_epoch,
            .eta_min = 3e-5,
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
┃ Accuracy                   ┃ 0.877800 ┃           0.877800 ┃
┃ Precision                  ┃ 0.878442 ┃           0.878442 ┃
┃ Recall                     ┃ 0.877800 ┃           0.877800 ┃
┃ F1 score                   ┃ 0.877967 ┃           0.877967 ┃
┃ True positive rate         ┃ 0.877800 ┃           0.877800 ┃
┃ True negative rate         ┃ 0.986422 ┃           0.986422 ┃
┃ Top-1 error                ┃ 0.122200 ┃           0.122200 ┃
┃ Expected calibration error ┃ 0.039478 ┃           0.039478 ┃
┃ Maximum calibration error  ┃ 0.803993 ┃           0.803993 ┃
┃ Cohen's kappa              ┃ 0.864222 ┃           0.864222 ┃
┃ Log loss                   ┃ 0.466842 ┃           0.466842 ┃
┃ Brier score                ┃ 0.193210 ┃           0.193210 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━┛

*/