#include "../../../include/Thot.h"
#include <array>

namespace {

    constexpr std::array<std::array<std::uint8_t, 3>, 6> kClassPalette{{
        std::array<std::uint8_t, 3>{  80, 227, 194},  // Water
        std::array<std::uint8_t, 3>{ 245, 166,  35},  // raw ground
        std::array<std::uint8_t, 3>{ 222,  89, 127},  // Road
        std::array<std::uint8_t, 3>{ 208,   2,  27},  // Building
        std::array<std::uint8_t, 3>{  65, 117,   5},  // Vegetation
        std::array<std::uint8_t, 3>{ 155, 155, 155},  // Unlabeled
    }};

    torch::Tensor ConvertRgbMasksToOneHot(const torch::Tensor& masks) {
        TORCH_CHECK(masks.dim() == 4, "Expected [B, 3, H, W]");
        TORCH_CHECK(masks.size(1) == 3, "RGB masks must have 3 channels");

        auto uint8_masks = masks.to(torch::kUInt8).contiguous();  // [B, 3, H, W]
        const auto B = uint8_masks.size(0);
        const auto H = uint8_masks.size(2);
        const auto W = uint8_masks.size(3);
        const auto C = static_cast<std::int64_t>(kClassPalette.size());

        // [B, 3, H, W] -> [B, H, W, 3] -> [N, 3] with N = B*H*W
        auto masks_flat = uint8_masks.permute({0, 2, 3, 1}).reshape({B * H * W, 3});

        std::vector<std::uint8_t> palette_vec;
        palette_vec.reserve(C * 3);
        for (const auto& rgb : kClassPalette) {
            palette_vec.push_back(rgb[0]); // R
            palette_vec.push_back(rgb[1]); // G
            palette_vec.push_back(rgb[2]); // B
        }

        auto palette = torch::tensor(
                            palette_vec,
                            torch::TensorOptions().dtype(torch::kUInt8)
                        )
                        .view({C, 3})
                        .to(uint8_masks.device());

        auto masks_i16   = masks_flat.to(torch::kInt16).unsqueeze(1);
        auto palette_i16 = palette.to(torch::kInt16).unsqueeze(0);

        auto diff  = masks_i16 - palette_i16;
        auto dist2 = diff.mul(diff).sum(-1);

        auto min_idx = std::get<1>(dist2.min(1));
        auto one_hot_flat = torch::nn::functional::one_hot(min_idx, C).to(torch::kFloat32);
        auto one_hot = one_hot_flat.view({B, H, W, C}).permute({0, 3, 1, 2});     // [B,C,H,W]

        return one_hot;
    }

    struct CustomTrainingOptions {
        std::size_t epochs{20};
        std::size_t batch_size{8};
        double dice_weight{0.7};
        double bce_weight{0.3};
    };

    void Train(Thot::Model& model, torch::Tensor inputs, torch::Tensor targets, const CustomTrainingOptions& options) {
        const auto total_samples = inputs.size(0);

        auto pinned_inputs  = Thot::async_pin_memory(inputs.contiguous());
        auto pinned_targets = Thot::async_pin_memory(targets.contiguous());
        auto host_inputs    = pinned_inputs.materialize();
        auto host_targets   = pinned_targets.materialize();

        const auto device      = model.device();
        const auto batch_size  = static_cast<std::int64_t>(options.batch_size);
        auto dice_descriptor   = Thot::Loss::Dice();

        model.train();

        for (std::size_t epoch = 0; epoch < options.epochs; ++epoch) {
            double accumulated_total = 0.0;
            double accumulated_dice  = 0.0;
            double accumulated_bce   = 0.0;
            std::size_t batches      = 0;

            for (std::int64_t offset = 0; offset < total_samples; offset += batch_size) {
                const auto remaining     = total_samples - offset;
                const auto current_batch = std::min<std::int64_t>(batch_size, remaining);
                if (current_batch <= 0) break;

                auto batch_inputs = host_inputs.narrow(0, offset, current_batch).to(device, host_inputs.scalar_type(), /*non_blocking=*/true);
                auto batch_targets = host_targets.narrow(0, offset, current_batch).to(device, host_targets.scalar_type(), /*non_blocking=*/true);

                model.zero_grad();
                auto predictions   = model.forward(batch_inputs);
                batch_targets      = batch_targets.to(predictions.scalar_type());

                auto probs = torch::softmax(predictions, /*dim=*/1);
                auto dice_loss = Thot::Loss::Details::compute(dice_descriptor, probs, batch_targets);

                auto clamped_predictions = predictions.clamp(1e-7, 1.0 - 1e-7);
                auto bce_loss = torch::nn::functional::binary_cross_entropy_with_logits(
                    predictions, // raw logits
                    batch_targets, // one-hot [B, 6, H, W] float
                    torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kMean));


                auto total_loss = dice_loss.mul(options.dice_weight)
                                + bce_loss.mul(options.bce_weight);

                total_loss.backward();
                model.step();

                accumulated_total += total_loss.detach().item<double>();
                accumulated_dice  += dice_loss.detach().item<double>();
                accumulated_bce   += bce_loss.detach().item<double>();
                ++batches;
            }

            if (batches > 0) {
                accumulated_total /= static_cast<double>(batches);
                accumulated_dice  /= static_cast<double>(batches);
                accumulated_bce   /= static_cast<double>(batches);
            }

            std::cout << "Epoch " << (epoch + 1) << "/" << options.epochs
                      << " | [Loss]: Dice(" << accumulated_dice  << ")"
                      << " + BCE(" << accumulated_bce   << ")"
                      << " = " << accumulated_total << std::endl;
        }

        host_inputs    = torch::Tensor();
        host_targets   = torch::Tensor();
        pinned_inputs  = Thot::AsyncPinnedTensor();
        pinned_targets = Thot::AsyncPinnedTensor();
    }

    torch::Tensor ColorizeClassMasks(const torch::Tensor& class_indices) {
        // class_indices: [B, H, W] with values in [0, num_classes)
        TORCH_CHECK(class_indices.dim() == 3, "Expected [B, H, W] for class_indices");

        const auto B = class_indices.size(0);
        const auto H = class_indices.size(1);
        const auto W = class_indices.size(2);

        auto options = torch::TensorOptions().dtype(torch::kUInt8).device(class_indices.device());

        auto rgb = torch::zeros({B, 3, H, W}, options);

        for (std::size_t c = 0; c < kClassPalette.size(); ++c) {
            const auto& color = kClassPalette[c]; // RGB

            auto mask = (class_indices == static_cast<std::int64_t>(c));

            for (int ch = 0; ch < 3; ++ch) {
                auto channel = rgb.select(1, ch);
                channel.masked_fill_(mask, color[ch]);
            }
        }
        return rgb.to(torch::kFloat32).div_(255.0f);
    }

}

void PrintUniqueColors(const torch::Tensor& batch, std::size_t idx) {
    TORCH_CHECK(batch.dim() == 4, "Expected [B, 3, H, W]");
    TORCH_CHECK(batch.size(1) == 3, "Expected 3 channels (RGB)");
    TORCH_CHECK(idx < static_cast<std::size_t>(batch.size(0)), "idx out of range");

    // [3, H, W] on CPU
    auto img = batch.index({static_cast<long>(idx)})
                    .detach()
                    .to(torch::kCPU)
                    .contiguous(); // [3, H, W]

    // If it's float, map to uint8
    if (img.scalar_type() == torch::kFloat32 || img.scalar_type() == torch::kFloat64) {
        // Your PNG loader with normalize=false is almost certainly 0–255 already,
        // but we clamp & convert just to be safe.
        img = img.clamp(0.0, 255.0).round().to(torch::kUInt8);
    } else if (img.scalar_type() != torch::kUInt8) {
        img = img.to(torch::kUInt8);
    }

    // [3, H, W] -> [H, W, 3] -> [N, 3]
    auto img_flat = img.permute({1, 2, 0})   // [H, W, 3]
                        .reshape({-1, 3});   // [N, 3]

    // Unique RGB triplets along dim 0
    auto unique = std::get<0>(
        torch::unique_dim(img_flat,
                          /*dim=*/0,
                          /*sorted=*/true,
                          /*return_inverse=*/false,
                          /*return_counts=*/false)
    ); // [K, 3]

    std::cout << "Unique colors in sample " << idx << ": " << unique.size(0) << "\n";
    for (int64_t i = 0; i < unique.size(0); ++i) {
        auto c = unique[i];
        auto r = c[0].item<int64_t>();
        auto g = c[1].item<int64_t>();
        auto b = c[2].item<int64_t>();
        std::cout << "  RGB(" << r << ", " << g << ", " << b << ")\n";
    }
}
//TODO:
//- Thot::Data::Load::Universal() with images reader manipulate color with .size= (compression color jitter)


int main() {
    Thot::Model model("");

    auto [x1, y1, x2, y2] =
        Thot::Data::Load::Universal("/home/moonfloww/Projects/DATASETS/Image/Satellite/DubaiSegmentationImages/Tile 1/",
        Thot::Data::Type::JPG{"images", {.grayscale = false, .normalize = false, .size_to_max_tile=false, /*.size=std::array<long,2>{256, 256},*/ .color_order = "RGB"}},
        Thot::Data::Type::PNG{"masks", {.normalize = false, /*.size=std::array<long,2>{256, 256},*/.size_to_max_tile=true, .color_order = "RGB"}}, {.train_fraction = .8f, .test_fraction = .2f, .shuffle = false});
    Thot::Plot::Data::Image(y1, {0});
    PrintUniqueColors(y1, 0);
    Thot::Data::Check::Size(x1, "Inputs Raw");
    Thot::Data::Check::Size(y1, "Outputs Raw");
    x1 = Thot::Data::Transform::Format::Downsample(x1, {.size=std::vector<int>{256, 256}});
    y1 = Thot::Data::Transform::Format::Downsample(y1, {.size=std::vector<int>{256, 256}});
    x2 = Thot::Data::Transform::Format::Downsample(x2, {.size=std::vector<int>{256, 256}});
    y2 = Thot::Data::Transform::Format::Downsample(y2, {.size=std::vector<int>{256, 256}});
    Thot::Data::Check::Size(x1, "Inputs Resized");
    Thot::Data::Check::Size(y1, "Outputs Resized");

    y1 = ConvertRgbMasksToOneHot(y1);
    y2 = ConvertRgbMasksToOneHot(y2);
    Thot::Data::Check::Size(y1, "Train Targets One-hot");
    std::tie(x1, y1) = Thot::Data::Transform::Augmentation::Flip(x1, y1, {.axes = {"x"}, .frequency = 1.f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transform::Augmentation::Flip(x1, y1, {.axes = {"y"}, .frequency = 0.5f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, true, false});
    std::tie(x1, y1) = Thot::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, false, false});
    std::tie(x1, y1) = Thot::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, false, false});
    std::tie(x1, y1) = Thot::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, false, false});
    std::tie(x1, y1) = Thot::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, false, false});
    std::tie(x1, y1) = Thot::Data::Transform::Augmentation::CLAHE(x1, y1, {.frequency = 1.f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transform::Augmentation::OpticalDistortion(x1, y1, {.frequency = 1.f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transform::Augmentation::AtmosphericDrift(x1, y1, {.frequency = 0.3f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transform::Augmentation::SunAngleJitter(x1, y1, {.frequency = 0.3f, .data_augment = true});

    //Take random 9 samples
    const auto n_samples = x1.size(0);
    auto perm = torch::randperm(n_samples, torch::TensorOptions().dtype(torch::kLong));
    auto rand9 = perm.narrow(0, 0, 9);
    std::vector<std::size_t> idx; idx.reserve(rand9.size(0));
    auto* data_ptr = rand9.data_ptr<int64_t>();
    for (int i = 0; i < rand9.size(0); ++i)
        idx.push_back(static_cast<std::size_t>(data_ptr[i]));

    Thot::Data::Check::Size(x1, "Inputs Augmented");
    Thot::Data::Check::Size(y1, "Targets Augmented");

    (void)Thot::Plot::Data::Image(x1, idx);
    auto y1_classes = y1.argmax(1);
    auto y1_rgb = ColorizeClassMasks(y1_classes);
    (void)Thot::Plot::Data::Image(y1_rgb, idx);
    //(void)Thot::Plot::Data::Image(y2, {0, 1});


    auto block = [&](int in_c, int out_c) {
        return Thot::Block::Sequential({
            Thot::Layer::Conv2d({in_c, out_c, {3,3}, {1,1}, {1,1}}, Thot::Activation::SiLU),
            Thot::Layer::Conv2d({out_c, out_c, {3,3}, {1,1}, {1,1}}, Thot::Activation::SiLU),
        });
    };

    auto upblock = [&](int in_c, int out_c) {
        return Thot::Block::Sequential({
            Thot::Layer::Upsample({.scale = {2,2}, .mode = Thot::UpsampleMode::Bilinear}),
            Thot::Layer::Conv2d({in_c, out_c, {3,3}, {1,1}, {1,1}}, Thot::Activation::SiLU),
        });
    };

    // encoders
    model.add(block(3, 64), "enc1");
    model.add(Thot::Layer::MaxPool2d({{2,2},{2,2}}));
    model.add(block(64, 64), "enc2");
    model.add(Thot::Layer::MaxPool2d({{2,2},{2,2}}));

    // decoders
    model.add(upblock(64, 64), "dec1");
    model.add(block(64, 64), "dec_block1");
    model.add(upblock(64, 32), "dec2");
    model.add(block(32, 32), "dec_block2");
    model.add(block(32, 32), "dec_block3");
    model.add(block(32, 32), "dec_block4");
    model.add(block(32, 32), "dec_block5");

    model.add(Thot::Layer::Conv2d({32, 6, {1,1}, {1,1}, {0,0}}, Thot::Activation::Identity));


    model.set_loss(Thot::Loss::Dice());
    model.set_optimizer(Thot::Optimizer::AdamW({.learning_rate=1e-4, .weight_decay = 2e-4}));


    //Custom loop for dual-loss
    CustomTrainingOptions training_options{};
    training_options.epochs = 20;
    training_options.batch_size = 12;
    training_options.dice_weight = 0.6;
    training_options.bce_weight = 1-training_options.dice_weight;

    const auto total_training_samples = x1.size(0);
    const auto steps_per_epoch = static_cast<std::size_t>((total_training_samples + training_options.batch_size - 1) / training_options.batch_size);
    const auto total_training_steps = std::max<std::size_t>(1, training_options.epochs * std::max<std::size_t>(steps_per_epoch, 1));

    model.set_loss(Thot::Loss::Dice());
    model.set_optimizer(
        Thot::Optimizer::AdamW({.learning_rate = 1e-4, .weight_decay = 1e-4}),
        Thot::LrScheduler::CosineAnnealing({
            .T_max = (total_training_steps),
            .eta_min = 1e-6,
            .warmup_steps = std::min<std::size_t>(steps_per_epoch * 5, total_training_steps / 5),
            .warmup_start_factor = 0.1,
        }));

    model.set_regularization({Thot::Regularization::SWAG({
        .coefficient = 5e-4,
        .variance_epsilon = 1e-6,
        .start_step = static_cast<std::size_t>(0.65 * static_cast<double>(total_training_steps)),
        .accumulation_stride = std::max<std::size_t>(1, steps_per_epoch),
        .max_snapshots = 20,
    })});

    model.use_cuda(torch::cuda::is_available());
    Train(model, x1, y1, training_options);

    torch::NoGradGuard guard;
    Thot::Data::Check::Size(x2, "Test Inputs");
    Thot::Data::Check::Size(y2, "Test Targets");
    model.evaluate(x2, y2, Thot::Evaluation::Segmentation, {
        Thot::Metric::Classification::Accuracy,
        Thot::Metric::Classification::Precision,
        Thot::Metric::Classification::Recall,
        Thot::Metric::Classification::JaccardIndexMicro,
        Thot::Metric::Classification::HausdorffDistance,
        Thot::Metric::Classification::BoundaryIoU,
    },
    {.batch_size = 12, .buffer_vram=2});

    {
        model.eval();

        const auto n_val = x2.size(0);
        const auto num_samples_to_plot = std::min<std::int64_t>(4, n_val);
        TORCH_CHECK(num_samples_to_plot > 0, "No validation samples available for plotting");

        auto perm = torch::randperm(n_val, torch::TensorOptions().dtype(torch::kLong));
        auto idx  = perm.narrow(0, 0, num_samples_to_plot); // [4]

        std::vector<std::size_t> local_indices(static_cast<std::size_t>(num_samples_to_plot));
        std::iota(local_indices.begin(), local_indices.end(), 0);

        const auto device = model.device();

        auto inputs_batch  = x2.index_select(0, idx).to(device);
        auto targets_batch = y2.index_select(0, idx).to(device);

        auto preds = model.forward(inputs_batch);

        auto preds_cpu   = preds.detach().to(torch::kCPU);
        auto targets_cpu = targets_batch.detach().to(torch::kCPU);

        auto pred_classes = preds_cpu.argmax(1);
        auto target_classes = targets_cpu.argmax(1);

        auto pred_rgb = ColorizeClassMasks(pred_classes).contiguous();
        auto target_rgb = ColorizeClassMasks(target_classes);

        Thot::Data::Check::Size(pred_rgb, "RGB Test Target");
        Thot::Data::Check::Size(target_rgb, "RGB Test Output");
        Thot::Data::Check::Size(pred_classes, "Hot-One Test Target");
        Thot::Data::Check::Size(target_classes, "Hot-One Test Output");
        (void)Thot::Plot::Data::Image(target_rgb, local_indices);
        (void)Thot::Plot::Data::Image(pred_rgb,   local_indices);
    }

    return 0;
}