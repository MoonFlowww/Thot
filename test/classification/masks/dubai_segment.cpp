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

                auto dice_loss = Thot::Loss::Details::compute(
                    dice_descriptor, predictions, batch_targets);

                auto clamped_predictions = predictions.clamp(1e-7, 1.0 - 1e-7);
                auto bce_loss = torch::nn::functional::binary_cross_entropy(
                    clamped_predictions,
                    batch_targets,
                    torch::nn::functional::BinaryCrossEntropyFuncOptions()
                        .reduction(torch::kMean));

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
                      << " + BCE("        << accumulated_bce   << ")"
                      << " = "            << accumulated_total << std::endl;
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

        auto options = torch::TensorOptions()
                           .dtype(torch::kUInt8)
                           .device(class_indices.device());

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


int main() {
    Thot::Model model("");

    auto [x1, y1, x2, y2] =
        Thot::Data::Load::Universal("/home/moonfloww/Projects/DATASETS/Image/Satellite/DubaiSegmentationImages/Tile 1/",
            Thot::Data::Type::JPG{"images", {.grayscale = false, .normalize = true, .pad_to_max_tile=true, .color_order = "RGB"}},
            Thot::Data::Type::PNG{"masks", {.normalize = false, .pad_to_max_tile=true, .color_order = "RGB"}}, {.train_fraction = .8f, .test_fraction = .2f, .shuffle = true});

    Thot::Data::Check::Size(x1, "Inputs Raw");
    Thot::Data::Check::Size(y1, "Outputs Raw");
    (void)Thot::Plot::Data::Image(x1, {0});
    x1 = Thot::Data::Transform::Format::Downscale(x1, {.targetsize = std::array<int64_t, 2>{256, 256}});
    y1 = Thot::Data::Transform::Format::Downscale(y1, {.targetsize = std::array<int64_t, 2>{256, 256}});
    x2 = Thot::Data::Transform::Format::Downscale(x2, {.targetsize = std::array<int64_t, 2>{256, 256}});
    y2 = Thot::Data::Transform::Format::Downscale(y2, {.targetsize = std::array<int64_t, 2>{256, 256}});
    Thot::Data::Check::Size(x1, "Inputs Resized");
    Thot::Data::Check::Size(y1, "Outputs Resized");
    (void)Thot::Plot::Data::Image(x1, {0});

    std::tie(x1, y1) = Thot::Data::Transforms::Augmentation::Flip(x1, y1, {.axes = {"x"}, .frequency = 1.f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transforms::Augmentation::Flip(x1, y1, {.axes = {"y"}, .frequency = 0.5f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, 0, 1.f, true, false});
    std::tie(x1, y1) = Thot::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {32, 32}, 0, 1.f, true, false});
    std::tie(x1, y1) = Thot::Data::Transforms::Augmentation::CLAHE(x1, y1, {.frequency = 1.f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transforms::Augmentation::OpticalDistortion(x1, y1, {.frequency = 1.f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transforms::Augmentation::AtmosphericDrift(x1, y1, {.frequency = 0.3f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transforms::Augmentation::SunAngleJitter(x1, y1, {.frequency = 0.3f, .data_augment = true});

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
    Thot::Data::Check::Size(y2, "test Targets");

    (void)Thot::Plot::Data::Image(x1, idx);
    //(void)Thot::Plot::Data::Image(y1, idx);
    //(void)Thot::Plot::Data::Image(y2, {0, 1});


    y1 = ConvertRgbMasksToOneHot(y1);
    y2 = ConvertRgbMasksToOneHot(y2);

    Thot::Data::Check::Size(y1, "Targets One-hot");



    const auto in_channels = x1.size(1);
    const auto image_height = x1.size(2);
    const auto image_width = x1.size(3);
    const auto mask_channels = y1.size(1);
    const auto mask_height = y1.size(2);
    const auto mask_width = y1.size(3);

    Thot::Block::Transformer::Vision::EncoderOptions vit_options{};
    vit_options.layers = 8;
    vit_options.embed_dim = 384;
    vit_options.attention.num_heads = 6;
    vit_options.attention.dropout = 0.3;
    vit_options.attention.batch_first = true;
    vit_options.attention.variant = Thot::Attention::Variant::Full;
    vit_options.feed_forward.mlp_ratio = 4.0;
    vit_options.feed_forward.activation = Thot::Activation::GeLU;
    vit_options.feed_forward.bias = true;
    vit_options.layer_norm.eps = 1e-6;
    vit_options.patch_embedding.in_channels = in_channels;
    vit_options.patch_embedding.embed_dim = vit_options.embed_dim;
    vit_options.patch_embedding.patch_size = 16;
    vit_options.patch_embedding.add_class_token = false;
    vit_options.patch_embedding.normalize = true;
    vit_options.patch_embedding.dropout = 0.2;
    vit_options.positional_encoding.type = Thot::Layer::Details::PositionalEncodingType::Learned;
    vit_options.positional_encoding.dropout = 0.2;
    vit_options.residual_dropout = 0.2;
    vit_options.attention_dropout = 0.2;
    vit_options.feed_forward_dropout = 0.2;
    vit_options.pre_norm = true;
    vit_options.final_layer_norm = true;


    const auto patch_size = vit_options.patch_embedding.patch_size;
    const auto tokens_h = (image_height - patch_size) / patch_size + 1;
    const auto tokens_w = (image_width - patch_size) / patch_size + 1;
    const auto patch_area = patch_size * patch_size;
    const auto patch_projection_dim = mask_channels * patch_area;

    //model.add(Thot::Layer::SoftDropout({.probability = 0.5, .noise_mean = 0.2, .noise_std = 1, .noise_type = Thot::Layer::SoftDropoutOptions::NoiseType::InterleavedGradientNoise}));
    model.add(Thot::Block::Transformer::Vision::Encoder(vit_options));
    model.add(Thot::Layer::HardDropout({ .probability = 0.1 }));
    model.add(Thot::Layer::FC({vit_options.embed_dim, patch_projection_dim, true}, Thot::Activation::Identity, Thot::Initialization::XavierUniform));
    model.add(Thot::Layer::PatchUnembed(
{.channels = mask_channels, .tokens_height = tokens_h, .tokens_width = tokens_w, .patch_size = patch_size, .target_height = mask_height, .target_width = mask_width, .align_corners = false },
        Thot::Activation::Sigmoid));

    model.set_loss(Thot::Loss::Dice());
    model.set_optimizer(Thot::Optimizer::AdamW({.learning_rate=1e-4, .weight_decay = 2e-4}));


    //Custom loop for dual-loss
    CustomTrainingOptions training_options{};
    training_options.epochs = 15;
    training_options.batch_size = 16;
    training_options.dice_weight = 0.6;
    training_options.bce_weight = 1-training_options.dice_weight;

    const auto total_training_samples = x1.size(0);
    const auto steps_per_epoch = static_cast<std::size_t>((total_training_samples + training_options.batch_size - 1) / training_options.batch_size);
    const auto total_training_steps = std::max<std::size_t>(1, training_options.epochs * std::max<std::size_t>(steps_per_epoch, 1));

    model.set_loss(Thot::Loss::Dice());
    model.set_optimizer(
        Thot::Optimizer::AdamW({.learning_rate = 1e-4, .weight_decay = 1e-4}),
        Thot::LrScheduler::CosineAnnealing({
            .T_max = total_training_steps,
            .eta_min = 1e-6,
            .warmup_steps = std::min<std::size_t>(steps_per_epoch * 5, total_training_steps / 5),
            .warmup_start_factor = 0.1,
        }));

    model.set_regularization({Thot::Regularization::SWAG({
        .coefficient = 5e-4,
        .variance_epsilon = 1e-6,
        .start_step = static_cast<std::size_t>(0.8 * static_cast<double>(total_training_steps)),
        .accumulation_stride = std::max<std::size_t>(1, steps_per_epoch),
        .max_snapshots = 20,
    })});

    model.use_cuda(torch::cuda::is_available());
    Train(model, x1, y1, training_options);

    torch::NoGradGuard guard;

    model.evaluate(x2, y2, Thot::Evaluation::Classification, {
        Thot::Metric::Classification::Accuracy,
        Thot::Metric::Classification::Precision,
        Thot::Metric::Classification::Recall,
        Thot::Metric::Classification::JaccardIndexMicro,
        Thot::Metric::Classification::HausdorffDistance,
        Thot::Metric::Classification::BoundaryIoU,
    },
    {.batch_size = 8, .buffer_vram=2});

    {
        model.eval();

        const auto n_val = x2.size(0);
        const auto num_samples_to_plot = std::min<std::int64_t>(4, n_val);
        TORCH_CHECK(num_samples_to_plot > 0, "No validation samples available for plotting");

        auto perm = torch::randperm(n_val, torch::TensorOptions().dtype(torch::kLong));
        auto idx  = perm.narrow(0, 0, num_samples_to_plot); // [4]

        // local indices 0..3 for plotting the small batch
        std::vector<std::size_t> local_indices(static_cast<std::size_t>(num_samples_to_plot));
        std::iota(local_indices.begin(), local_indices.end(), 0);

        const auto device = model.device();

        auto inputs_batch  = x2.index_select(0, idx).to(device);
        auto targets_batch = y2.index_select(0, idx).to(device);

        auto preds = model.forward(inputs_batch);

        auto preds_cpu   = preds.detach().to(torch::kCPU);
        auto targets_cpu = targets_batch.detach().to(torch::kCPU);

        // From one-hot [B, C, H, W] → class indices [B, H, W]
        auto pred_classes = preds_cpu.argmax(1);
        auto target_classes = targets_cpu.argmax(1);

        auto pred_rgb = ColorizeClassMasks(pred_classes).contiguous();
        auto target_rgb = ColorizeClassMasks(target_classes);

        Thot::Data::Check::Size(pred_rgb, "Test Target");
        Thot::Data::Check::Size(target_rgb, "Test Output");
        (void)Thot::Plot::Data::Image(target_rgb, local_indices); // Ground truth masks
        (void)Thot::Plot::Data::Image(pred_rgb,   local_indices); // Predicted masks

        // If you also want the input tiles next to them:
        auto inputs_cpu = inputs_batch.detach().to(torch::kCPU);
        (void)Thot::Plot::Data::Image(inputs_cpu, local_indices);
    }

    return 0;
}