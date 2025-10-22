//#include "../include/libthot.h"
//#include <iostream>

//#include "../src/data/dimreduction.hpp"

int instance() {
    Thot::Network model;
    model.choose_device(true);


    Thot::Block::Transformer::Classic::EncoderOptions encoder{};
    encoder.layers = 2;
    encoder.attention.num_heads = 4;
    encoder.attention.variant = Thot::Attention::Variant::Full;
    encoder.feed_forward.mlp_ratio = 2.0;
    encoder.feed_forward.activation = Thot::Activation::GELU;
    encoder.layer_norm.elementwise_affine = false;
    encoder.positional_encoding.type = Thot::Block::Transformer::Classic::PositionalEncodingType::Sinusoidal;
    encoder.positional_encoding.dropout = 0.05;
    encoder.dropout = 0.1;
    model.block(Thot::Block::Transformer::Classic::Encoder(encoder));

    Thot::Block::Transformer::Classic::DecoderOptions decoder{};
    decoder.layers = 2;
    decoder.self_attention.num_heads = 4;
    decoder.self_attention.variant = Thot::Attention::Variant::Full;
    decoder.cross_attention.num_heads = 4;
    decoder.feed_forward.mlp_ratio = 2.0;
    decoder.layer_norm.eps = 1e-4;
    decoder.positional_encoding.type = Thot::Block::Transformer::Classic::PositionalEncodingType::Learned;
    decoder.dropout = 0.05;
    model.block(Thot::Block::Transformer::Classic::Decoder(decoder));




    model.block(Thot::Block::Sequential({
        Thot::Layer::Conv2d(
            {3, 64, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false},
            Thot::Activation::Linear,
            Thot::Initialization::HeNormal
        ),
        Thot::Layer::BatchNorm2d(
            {64, 1e-5, 0.1, true, true},
            Thot::Activation::SiLU
        ),
        Thot::Layer::Conv2d(
            {64, 64, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, true},
            Thot::Activation::Linear,
            Thot::Initialization::HeNormal,
            {}
        ),
        Thot::Layer::BatchNorm2d(
            {64, 1e-5, 0.1, true, true},
            Thot::Activation::SiLU
        ),
        Thot::Layer::MaxPool2d({{2, 2}, {2, 2}})
    }));

    model.add(Thot::Layer::Dropout({ .probability = 0.3 }));


    model.block(Thot::Block::Residual({
        Thot::Layer::Conv2d(
            {64, 64, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false},
            Thot::Activation::Linear,
            Thot::Initialization::HeNormal
        ),
        Thot::Layer::BatchNorm2d(
            {64, 1e-5, 0.1, true, true},
            Thot::Activation::SiLU
        ),
        Thot::Layer::Conv2d(
            {64, 64, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, true},
            Thot::Activation::Linear,
            Thot::Initialization::HeNormal
        )
    }, 3, {}, { .final_activation = Thot::Activation::SiLU }));

    model.block(Thot::Block::Residual({
        Thot::Layer::Conv2d(
            {64, 128, {3, 3}, {2, 2}, {1, 1}, {1, 1}, 1, false},
            Thot::Activation::Linear,
            Thot::Initialization::HeNormal
        ),
        Thot::Layer::BatchNorm2d(
            {128, 1e-5, 0.1, true, true},
            Thot::Activation::SiLU
        ),
        Thot::Layer::Conv2d(
            {128, 128, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, true},
            Thot::Activation::Linear,
            Thot::Initialization::HeNormal
        )
    }, 1, {}, { .final_activation = Thot::Activation::SiLU }));

    model.add(Thot::Layer::Dropout({ .probability = 0.3 }));

    model.block(Thot::Block::Sequential({
        Thot::Layer::Conv2d(
            {128, 256, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false},
            Thot::Activation::Linear,
            Thot::Initialization::HeNormal
        ),
        Thot::Layer::BatchNorm2d(
            {256, 1e-5, 0.1, true, true},
            Thot::Activation::SiLU
        ),
        Thot::Layer::AdaptiveAvgPool2d({{1, 1}})
    }));

    model.block(Thot::Block::Sequential({
        Thot::Layer::Conv2d(
            {256, 128, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false},
            Thot::Activation::Linear,
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
    model.add(Thot::Layer::Dropout({.probability = 0.5}));
    model.add(Thot::Layer::FC({512, 10, true}, Thot::Activation::Linear, Thot::Initialization::HeNormal));


    std::cout << "Loading CIFAR-10..." << std::endl;
    auto [train_inputs, train_targets, test_inputs, test_targets] = Thot::Data::Load::CIFAR10("/home/moonfloww/Projects/DATASETS/CIFAR10", 1.f, 1.f);

    auto augment_dataset = [](const torch::Tensor& inputs) {
        auto augmented = inputs.clone();
        const int64_t batch = augmented.size(0);
        const int64_t height = augmented.size(2);
        const int64_t width = augmented.size(3);
        const int64_t cutout_size = 12;

        for (int64_t index = 0; index < batch; ++index) {
            if (torch::rand({1}).item<double>() < 0.5) {
                auto sample = augmented[index];
                sample = Thot::Data::Manipulation::Flip(sample, {1}, true, 1.0, false);
                augmented.index_put_({index}, sample);
            }

            if (torch::rand({1}).item<double>() < 0.7) {
                const int64_t max_y = std::max<int64_t>(1, height - cutout_size + 1);
                const int64_t max_x = std::max<int64_t>(1, width - cutout_size + 1);
                const int64_t offset_y = torch::randint(0, max_y, {1}, torch::kLong).item<int64_t>();
                const int64_t offset_x = torch::randint(0, max_x, {1}, torch::kLong).item<int64_t>();
                auto sample = augmented[index];
                sample = Thot::Data::Manipulation::Cutout(sample, {offset_y, offset_x}, {cutout_size, cutout_size}, 0.0, true, 1.0, false);
                augmented.index_put_({index}, sample);
            }
        }

        return augmented;
    };
    std::cout << "\n\nData Augmentation:" << std::endl;
    auto augmented_train_inputs = augment_dataset(train_inputs);
    train_inputs = torch::cat({train_inputs, augmented_train_inputs}, 0);
    train_targets = torch::cat({train_targets, train_targets.clone()}, 0);


    std::cout << Thot::Data::Check::Imbalance(train_targets, test_targets).str() << std::endl;

    std::cout << Thot::Data::Check::Shuffled(train_targets) << std::endl;

    auto p = Thot::Data::Manipulation::Shuffle(train_inputs, train_targets);
    train_inputs = p.first;
    train_targets = p.second;

    std::cout << Thot::Data::Check::Shuffled(train_targets) << std::endl;


    Thot::TrainOptions train_opts{};
    train_opts.epochs = 120;
    train_opts.batch_size = 128;
    train_opts.test_fraction = 0.1;
    train_opts.restore_best_test_loss = true;

    const int64_t steps_per_epoch = (train_inputs.size(0) + train_opts.batch_size - 1) / train_opts.batch_size;

    model.set_optimizer(
        Thot::Optimizer::AdamW({
            .learning_rate = 1e-3,
            .weight_decay = 2e-2
        }),
        Thot::LrScheduler::CosineAnnealing({.T_max=train_opts.epochs * steps_per_epoch, .eta_min=5e-6, .warmup_steps=5*steps_per_epoch, .warmup_start_factor=0.1})
    );
    model.set_loss(Thot::Loss::CrossEntropy, {.label_smoothing = 0.1});

    //SWAG parameters
    const int64_t swag_start_epoch = 55;
    const int64_t swag_stride_epochs = 2;
    const std::size_t swag_start_step = static_cast<std::size_t>(std::max<int64_t>(0, swag_start_epoch * steps_per_epoch));
    const std::size_t swag_accumulation_stride = static_cast<std::size_t>( std::max<int64_t>(1, swag_stride_epochs * steps_per_epoch));
    const std::size_t swag_max_snapshots = 30;
    model.set_penalization({Thot::Penalization::SWAG(swag_start_step, swag_accumulation_stride, swag_max_snapshots)});


    std::cout << "\n\n\nStarting training with augmentation..." << std::endl;
    model.fit(train_inputs, train_targets, train_opts, Thot::KFold::Classic(1));


    model.rule({Thot::Algo::SSM::ELBO});
    //auto cx1= Thot::Data::Manipulation::Cutout(x1, {32, 32}, {8, 8}, 0.0, true, 0.3, false);
    //auto fcx1 = Thot::Data::Manipulation::Flip(cx1, {1}, true, 0.5, false);
    //auto ffcx1 = Thot::Data::Manipulation::Flip(cx1, {0}, true, 0.5, false);

    /*
    at::Tensor Ul1;
    {
        auto Gx1 = Thot::Data::Manipulation::Grayscale(x1); // Gray scale

        auto orig_sizes = Gx1.sizes();
        torch::Tensor* pTrainMatrix = new torch::Tensor(Gx1.reshape({Gx1.size(0), -1}));
        auto [low_rank1, sparse1] = Thot::Data::DimReduction::RPCA(*pTrainMatrix); // RPCA
        delete pTrainMatrix;
        low_rank1 = low_rank1.contiguous().reshape(orig_sizes);
        Ul1 = low_rank1;
        //Ul1 = Thot::Data::Manipulation::Upsample(low_rank1, {4.0}, torch::kBicubic, false, true); // UpSampling ->sqrt(x): 32 -> 128
    }
    */

    /*
    at::Tensor Ul2;
    {
        auto Gx2 = Thot::Data::Manipulation::Grayscale(x2); // Gray

        auto orig_sizes = Gx2.sizes();
        torch::Tensor* pTestMatrix = new torch::Tensor(Gx2.reshape({Gx2.size(0), -1}));
        auto [low_rank2, sparse2] = Thot::Data::DimReduction::RPCA(*pTestMatrix); // RPCA
        delete pTestMatrix;
        low_rank2 = low_rank2.contiguous().reshape(orig_sizes);

        Ul2 = low_rank2;

        //Ul2 = Thot::Data::Manipulation::Upsample(low_rank2, {4.0}, torch::kBicubic, false, true); // Upsample ->sqrt(x): 32 -> 128
    }
    */
    model.evaluate(test_inputs, test_targets, Thot::Evaluation::Classification,{
        Thot::Metric::Classification::Precision,
        Thot::Metric::Classification::Recall,
        Thot::Metric::Classification::F1,
        Thot::Metric::Classification::TruePositiveRate,
        Thot::Metric::Classification::TrueNegativeRate,
        Thot::Metric::Classification::Top1Accuracy,
        Thot::Metric::Classification::ExpectedCalibrationError,
        Thot::Metric::Classification::MaximumCalibrationError,
        Thot::Metric::Classification::CohensKappa,
        Thot::Metric::Classification::LogLoss,
        Thot::Metric::Classification::BrierScore,
        }, {.batch_size=128});

    model.stresstest(test_inputs, test_targets, 5, 15, 0.01, 0.99, true, 128);


    return 0;
}


/*



CIFAR10:

NoKfold + SWAG + AdamW:
Top1 Error max(1-Prec)c: 20.5%
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                        Test Metrics                        ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Initial test loss            ┃ 1.454159                    ┃
┃ Final test loss              ┃ 0.652729                    ┃
┃ Best test loss               ┃ 0.652729                    ┃
┃ Best test epoch              ┃ 119/120                     ┃
┃ Restored best model          ┃ yes                         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Evaluation: Classification                                 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Global metrics                                             ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━┫
┃ Metric                     ┃ Macro    ┃ Weighted (support) ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━┫
┃ Precision                  ┃ 0.900010 ┃           0.900010 ┃
┃ Recall                     ┃ 0.900000 ┃           0.900000 ┃
┃ F1 score                   ┃ 0.899924 ┃           0.899924 ┃
┃ True positive rate         ┃ 0.900000 ┃           0.900000 ┃
┃ True negative rate         ┃ 0.988889 ┃           0.988889 ┃
┃ Top-1 Loss                 ┃ 0.900000 ┃           0.900000 ┃
┃ Expected calibration error ┃ 0.070450 ┃           0.070450 ┃
┃ Maximum calibration error  ┃ 0.178262 ┃           0.178262 ┃
┃ Cohen's kappa              ┃ 0.888889 ┃           0.888889 ┃
┃ Log loss                   ┃ 0.403026 ┃           0.403026 ┃
┃ Brier score                ┃ 0.156863 ┃           0.156863 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━┛

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Per-class metrics                                                                                                                ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Metric             ┃ Label 0  ┃ Label 1  ┃ Label 2  ┃ Label 3  ┃ Label 4  ┃ Label 5  ┃ Label 6  ┃ Label 7  ┃ Label 8  ┃ Label 9  ┃
┣━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Support            ┃     1000 ┃     1000 ┃     1000 ┃     1000 ┃     1000 ┃     1000 ┃     1000 ┃     1000 ┃     1000 ┃     1000 ┃
┃ Precision          ┃ 0.897260 ┃ 0.945813 ┃ 0.877743 ┃ 0.795796 ┃ 0.891732 ┃ 0.832673 ┃ 0.932528 ┃ 0.949846 ┃ 0.942058 ┃ 0.934653 ┃
┃ Recall             ┃ 0.917000 ┃ 0.960000 ┃ 0.840000 ┃ 0.795000 ┃ 0.906000 ┃ 0.841000 ┃ 0.926000 ┃ 0.928000 ┃ 0.943000 ┃ 0.944000 ┃
┃ F1 score           ┃ 0.907023 ┃ 0.952854 ┃ 0.858457 ┃ 0.795398 ┃ 0.898810 ┃ 0.836816 ┃ 0.929252 ┃ 0.938796 ┃ 0.942529 ┃ 0.939303 ┃
┃ True positive rate ┃ 0.917000 ┃ 0.960000 ┃ 0.840000 ┃ 0.795000 ┃ 0.906000 ┃ 0.841000 ┃ 0.926000 ┃ 0.928000 ┃ 0.943000 ┃ 0.944000 ┃
┃ True negative rate ┃ 0.988333 ┃ 0.993889 ┃ 0.987000 ┃ 0.977333 ┃ 0.987778 ┃ 0.981222 ┃ 0.992556 ┃ 0.994556 ┃ 0.993556 ┃ 0.992667 ┃
┃ Log loss           ┃ 0.342359 ┃ 0.220547 ┃ 0.582939 ┃ 0.742391 ┃ 0.383873 ┃ 0.601750 ┃ 0.305606 ┃ 0.314912 ┃ 0.268050 ┃ 0.267827 ┃
┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┛

*/