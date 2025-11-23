#include "../../../include/Thot.h"
#include <array>
#include <iostream>
#include <iomanip>
#include <tuple>
namespace {

    constexpr std::array<std::array<std::uint8_t, 3>, 6> kClassPalette{{
        std::array<std::uint8_t, 3>{60, 16, 152},
        std::array<std::uint8_t, 3>{110,193, 228},
        std::array<std::uint8_t, 3>{132, 41, 246},
        std::array<std::uint8_t, 3>{155, 155, 155},
        std::array<std::uint8_t, 3>{226, 169, 41},
        std::array<std::uint8_t, 3>{254, 221, 58},
    }};

    torch::Tensor ConvertRgbMasksToOneHot(const torch::Tensor& masks) {
        TORCH_CHECK(masks.dim() == 4, "Expected [B, 3, H, W]");
        TORCH_CHECK(masks.size(1) == 3, "RGB masks must have 3 channels");

        auto uint8_masks = masks.to(torch::kUInt8).contiguous();  // [B, 3, H, W]
        const auto B = uint8_masks.size(0);
        const auto H = uint8_masks.size(2);
        const auto W = uint8_masks.size(3);
        const auto C = static_cast<std::int64_t>(kClassPalette.size());

        // [B, 3, H, W] -> [B, H, W, 3] -> [N, 3]
        auto masks_flat = uint8_masks.permute({0, 2, 3, 1}).reshape({B * H * W, 3});

        std::vector<std::uint8_t> palette_vec;
        palette_vec.reserve(C * 3);
        for (const auto& rgb : kClassPalette) {
            palette_vec.push_back(rgb[0]);
            palette_vec.push_back(rgb[1]);
            palette_vec.push_back(rgb[2]);
        }

        auto palette = torch::tensor(
                            palette_vec,
                            torch::TensorOptions().dtype(torch::kUInt8))
                        .view({C, 3})
                        .to(uint8_masks.device());
        auto masks_i32   = masks_flat.to(torch::kInt32).unsqueeze(1); // [N, 1, 3]
        auto palette_i32 = palette.to(torch::kInt32).unsqueeze(0);    // [1, C, 3]

        auto diff  = masks_i32 - palette_i32;               // [N, C, 3], int32
        auto dist2 = diff.mul(diff).sum(-1);                // [N, C],  int32

        auto min_idx = std::get<1>(dist2.min(1));           // [N]

        auto one_hot_flat = torch::nn::functional::one_hot(min_idx, C)
                                .to(torch::kFloat32);       // [N, C]
        auto one_hot = one_hot_flat.view({B, H, W, C})
                                   .permute({0, 3, 1, 2});  // [B, C, H, W]

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

                auto probs = predictions;//torch::softmax(predictions, /*dim=*/1);
                auto dice_loss = Thot::Loss::Details::compute(dice_descriptor, probs, batch_targets);

                //auto clamped_predictions = predictions.clamp(1e-7, 1.0 - 1e-7);
                auto bce_loss = torch::nn::functional::binary_cross_entropy_with_logits(
                    predictions, // raw logits
                    batch_targets, // one-hot [B, 6, H, W] float
                    torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kMean));


                auto total_loss = dice_loss.mul(options.dice_weight) + bce_loss.mul(options.bce_weight);

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

#include <iostream>
#include <iomanip>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cstdint>

void PrintUniqueColors(const torch::Tensor& batch, std::size_t idx) {
    TORCH_CHECK(batch.dim() == 4, "Expected [B, C, H, W]");
    TORCH_CHECK(idx < static_cast<std::size_t>(batch.size(0)), "idx out of range");

    // [C, H, W] on CPU
    auto img = batch.index({static_cast<long>(idx)})
                    .detach()
                    .to(torch::kCPU)
                    .contiguous();  // [C, H, W]

    const auto C = img.size(0);
    const auto H = img.size(1);
    const auto W = img.size(2);
    const auto total_pixels = static_cast<double>(H * W);

    if (C == 3) {
        // ================= RGB PATH =================
        auto img_rgb = img;
        if (img_rgb.is_floating_point()) {
            // Assume 0–255-like, clamp & round to bytes
            img_rgb = img_rgb.clamp(0.0, 255.0).round().to(torch::kUInt8);
        } else if (img_rgb.scalar_type() != torch::kUInt8) {
            img_rgb = img_rgb.to(torch::kUInt8);
        }

        auto acc = img_rgb.accessor<std::uint8_t, 3>(); // [3, H, W]

        // color -> count
        std::unordered_map<std::uint32_t, int64_t> color_counts;
        color_counts.reserve(static_cast<std::size_t>(H * W) / 4 + 1);

        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                std::uint8_t r = acc[0][y][x];
                std::uint8_t g = acc[1][y][x];
                std::uint8_t b = acc[2][y][x];
                // pack RGB into 24 bits of a 32-bit int
                std::uint32_t key =
                    static_cast<std::uint32_t>(r) |
                    (static_cast<std::uint32_t>(g) << 8) |
                    (static_cast<std::uint32_t>(b) << 16);
                ++color_counts[key];
            }
        }

        std::vector<std::pair<std::uint32_t, int64_t>> entries;
        entries.reserve(color_counts.size());
        for (const auto& kv : color_counts) {
            entries.emplace_back(kv.first, kv.second);
        }

        std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
                      return a.first < b.first;});

        std::cout << "Unique RGB colors in sample " << idx
                  << " (H=" << H << ", W=" << W << "): "
                  << entries.size() << "\n";

        std::cout << std::fixed << std::setprecision(4);
        for (const auto& [key, cnt] : entries) {
            std::uint8_t r = static_cast<std::uint8_t>( key        & 0xFFu);
            std::uint8_t g = static_cast<std::uint8_t>((key >> 8)  & 0xFFu);
            std::uint8_t b = static_cast<std::uint8_t>((key >> 16) & 0xFFu);
            double freq = static_cast<double>(cnt) / total_pixels;

            std::cout << "  RGB(" << static_cast<int>(r)
                      << ", "    << static_cast<int>(g)
                      << ", "    << static_cast<int>(b)
                      << ")  pixels=" << cnt
                      << "  (" << freq * 100.0 << "%)\n";
        }
    } else { // hot one
        auto classes = img.argmax(/*dim=*/0);           // [H, W]
        classes = classes.to(torch::kCPU).contiguous();
        if (classes.scalar_type() != torch::kLong) {
            classes = classes.to(torch::kLong);
        }

        auto flat = classes.view({-1}); // [N]
        const auto N = flat.size(0);
        auto ptr = flat.data_ptr<int64_t>();

        std::unordered_map<int64_t, int64_t> counts;
        counts.reserve(32);
        for (int64_t i = 0; i < N; ++i) {
            ++counts[ptr[i]];
        }

        std::vector<std::pair<int64_t, int64_t>> entries;
        entries.reserve(counts.size());
        for (const auto& kv : counts) {
            entries.emplace_back(kv.first, kv.second);
        }
        std::sort(entries.begin(), entries.end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first; // sort by class id
                  });

        std::cout << "Unique classes in sample " << idx
                  << " (C=" << C << ", H=" << H << ", W=" << W << "): "
                  << entries.size() << "\n";

        std::cout << std::fixed << std::setprecision(4);
        for (const auto& [cls, cnt] : entries) {
            double freq = static_cast<double>(cnt) / total_pixels;
            std::cout << "  class " << cls
                      << "  pixels=" << cnt
                      << "  (" << freq * 100.0 << "%)\n";
        }
    }
}


int main() {
    Thot::Model model("");

    auto [x1, y1, x2, y2] =
        Thot::Data::Load::Universal("/home/moonfloww/Projects/DATASETS/Image/Satellite/DubaiSegmentationImages/",//Tile 1/
        Thot::Data::Type::JPG{"images", {.grayscale = false, .normalize = false, .size_to_max_tile=false, .size={256, 256}, .color_order = "RGB"}},
        Thot::Data::Type::PNG{"masks", {.normalize = false, .size={256, 256}, .InterpolationMode = Thot::Data::Transform::Format::Options::InterpMode::Nearest, .color_order = "RGB"}}, {.train_fraction = .8f, .test_fraction = .2f, .shuffle = false});
    Thot::Plot::Data::Image(y1, {0});
    PrintUniqueColors(y1, 0);
    Thot::Data::Check::Size(x1, "Inputs Raw");
    Thot::Data::Check::Size(y1, "Outputs Raw");

    y1 = ConvertRgbMasksToOneHot(y1);
    y2 = ConvertRgbMasksToOneHot(y2);
    std::cout << "Hot One: " << std::endl;
    PrintUniqueColors(y1, 0);

    Thot::Data::Check::Size(y1, "Train Targets One-hot");
    std::tie(x1, y1) = Thot::Data::Transform::Augmentation::Flip(x1, y1, {.axes = {"x"}, .frequency = 1.f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transform::Augmentation::Flip(x1, y1, {.axes = {"y"}, .frequency = .5f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, true, false});
    std::tie(x1, y1) = Thot::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, false, false});
    std::tie(x1, y1) = Thot::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, false, false});
    std::tie(x1, y1) = Thot::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, false, false});
    std::tie(x1, y1) = Thot::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, {-1,-1,-1}, 1.f, false, false});
    std::tie(x1, y1) = Thot::Data::Transform::Augmentation::CLAHE(x1, y1, {.frequency = 1.f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transform::Augmentation::OpticalDistortion(x1, y1, {.frequency = 1.f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transform::Augmentation::AtmosphericDrift(x1, y1, {.frequency = .3f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transform::Augmentation::SunAngleJitter(x1, y1, {.frequency = .3f, .data_augment = true});

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
    //(void)Thot::Plot::Data::Image(y1_rgb, idx);
    //(void)Thot::Plot::Data::Image(y2, {0, 1});


    auto block = [&](int in_c, int out_c) {
        return Thot::Block::Sequential({
            Thot::Layer::Conv2d({in_c, out_c, {3,3}, {1,1}, {1,1}}, Thot::Activation::GeLU, Thot::Initialization::HeUniform),
            Thot::Layer::Conv2d({out_c, out_c, {3,3}, {1,1}, {1,1}}, Thot::Activation::GeLU, Thot::Initialization::HeUniform),
        });
    };

    auto upblock = [&](int in_c, int out_c) {
        return Thot::Block::Sequential({
            Thot::Layer::Upsample({.scale = {2,2}, .mode = Thot::UpsampleMode::Bicubic}),
            Thot::Layer::Conv2d({in_c, out_c, {3,3}, {1,1}, {1,1}}, Thot::Activation::GeLU, Thot::Initialization::HeUniform),
        });
    };

    // encoders
    model.add(block(3, 64), "enc1");
    model.add(Thot::Layer::MaxPool2d({{2, 2}, {2, 2}}), "pool1");
    model.add(block(64, 128), "enc2");
    model.add(Thot::Layer::MaxPool2d({{2, 2}, {2, 2}}), "pool2");
    model.add(block(128, 256), "enc3");
    model.add(Thot::Layer::MaxPool2d({{2, 2}, {2, 2}}), "pool3");
    model.add(block(256, 512), "bottleneck");

    // decoders
    model.add(upblock(512, 256), "up3");
    model.add(block(512, 256), "dec3");
    model.add(upblock(256, 128), "up2");
    model.add(block(256, 128), "dec2");
    model.add(upblock(128, 64), "up1");
    model.add(block(128, 64), "dec1");

    model.add(Thot::Layer::Conv2d({64, 6, {1, 1}, {1, 1}, {0, 0}}, Thot::Activation::Identity), "logits");

    model.links({
        // encoder path
        Thot::LinkSpec{Thot::Port::Input("@input"), Thot::Port::Module("enc1")},
        Thot::LinkSpec{Thot::Port::Module("enc1"), Thot::Port::Module("pool1")},
        Thot::LinkSpec{Thot::Port::Module("pool1"), Thot::Port::Module("enc2")},
        Thot::LinkSpec{Thot::Port::Module("enc2"), Thot::Port::Module("pool2")},
        Thot::LinkSpec{Thot::Port::Module("pool2"), Thot::Port::Module("enc3")},
        Thot::LinkSpec{Thot::Port::Module("enc3"), Thot::Port::Module("pool3")},
        Thot::LinkSpec{Thot::Port::Module("pool3"), Thot::Port::Module("bottleneck")},

        // decoder with skip
        Thot::LinkSpec{Thot::Port::Module("bottleneck"), Thot::Port::Module("up3")},
        Thot::LinkSpec{Thot::Port::Join({"up3", "enc3"}, Thot::MergePolicy::Stack), Thot::Port::Module("dec3")},
        Thot::LinkSpec{Thot::Port::Module("dec3"), Thot::Port::Module("up2")},
        Thot::LinkSpec{Thot::Port::Join({"up2", "enc2"}, Thot::MergePolicy::Stack), Thot::Port::Module("dec2")},
        Thot::LinkSpec{Thot::Port::Module("dec2"), Thot::Port::Module("up1")},
        Thot::LinkSpec{Thot::Port::Join({"up1", "enc1"}, Thot::MergePolicy::Stack), Thot::Port::Module("dec1")},

        // head
        Thot::LinkSpec{Thot::Port::Module("dec1"), Thot::Port::Module("logits")},
        Thot::LinkSpec{Thot::Port::Module("logits"), Thot::Port::Output("@output")},
    }, true);

    //Custom loop for dual-loss
    CustomTrainingOptions training_options{};
    training_options.epochs =10;
    training_options.batch_size = 8;
    training_options.dice_weight = 0; // 0.6
    training_options.bce_weight = 1-training_options.dice_weight;

    const auto total_training_samples = x1.size(0);
    const auto steps_per_epoch = static_cast<std::size_t>((total_training_samples + training_options.batch_size - 1) / training_options.batch_size);
    const auto total_training_steps = std::max<std::size_t>(1, training_options.epochs * std::max<std::size_t>(steps_per_epoch, 1));

    model.set_loss(Thot::Loss::BCEWithLogits());
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
    std::cout << "PreTrain Targets: " << std::endl;
    PrintUniqueColors(y1, 0);
    //Train(model, x1, y1, training_options);
    model.train(x1, y1, {.epoch = 10, .batch_size = 8, .restore_best_state = true, .test = std::vector<at::Tensor>{x2, y2}, .graph_mode = Thot::GraphMode::Capture, .enable_amp = true});
    torch::NoGradGuard guard;
    Thot::Data::Check::Size(x2, "Test Inputs");
    Thot::Data::Check::Size(y2, "Test Targets");
    model.evaluate(x2, y2, Thot::Evaluation::Segmentation, {
        Thot::Metric::Classification::Accuracy,
        Thot::Metric::Classification::Precision,
        Thot::Metric::Classification::Recall,
        Thot::Metric::Classification::JaccardIndexMicro,
        //Thot::Metric::Classification::HausdorffDistance,
        //Thot::Metric::Classification::BoundaryIoU,
    },
    {.batch_size = 8, .buffer_vram=2});

    {
        model.eval();

        const auto device = model.device();

        auto inputs_batch  = x2.index({0}).unsqueeze(0).to(device);
        auto targets_batch = y2.index({0}).unsqueeze(0).to(device);

        auto preds = model.forward(inputs_batch);
        Thot::Data::Check::Size(preds, "Fresh Network Prediction");
        auto preds_cpu   = preds.detach().to(torch::kCPU);
        auto targets_cpu = targets_batch.detach().to(torch::kCPU);

        auto pred_classes = preds_cpu.argmax(1); // move 2nd dim (6 labels) into a single one
        auto target_classes = targets_cpu.argmax(1);

        auto pred_rgb = ColorizeClassMasks(pred_classes);
        auto target_rgb = ColorizeClassMasks(target_classes);

        Thot::Data::Check::Size(pred_rgb, "Hot-One Test Target");
        Thot::Data::Check::Size(target_rgb, "Hot-One Test Output");
        Thot::Data::Check::Size(pred_rgb, "RGB Test Target");
        Thot::Data::Check::Size(target_rgb, "RGB Test Output");

        (void)Thot::Plot::Data::Image(target_rgb, {0});
        (void)Thot::Plot::Data::Image(pred_rgb,   {0});

        std::cout << "Y2:" << std::endl;
        PrintUniqueColors(y2, 0);

        std::cout << "Forecast:" << std::endl;
        PrintUniqueColors(preds_cpu, 0);

        auto preds0 = preds_cpu.index({0});
        const auto C = preds0.size(0);
        for (int64_t c = 0; c < C; ++c) {
            auto ch = preds0.index({c});
            auto ch_min  = ch.min().item<float>();
            auto ch_max  = ch.max().item<float>();
            auto ch_mean = ch.mean().item<float>();
            std::cout << "Channel " << c
                      << "  min="  << ch_min
                      << "  max="  << ch_max
                      << "  mean=" << ch_mean
                      << std::endl;
        }


        std::cout << "Target:" << std::endl;
        PrintUniqueColors(targets_cpu, 0);
    }

    return 0;
}