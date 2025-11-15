#include "../../../include/Thot.h"


namespace {

    struct CustomTrainingOptions {
        std::size_t epochs{20};
        std::size_t batch_size{8};
        double dice_weight{0.7};
        double bce_weight{0.3};
    };

    void Train(Thot::Model& model, torch::Tensor inputs, torch::Tensor targets, const CustomTrainingOptions& options) {
        if (!model.has_optimizer()) {
            throw std::logic_error("Custom training requires the model to have an optimizer configured.");
        }
        if (options.batch_size == 0) {
            throw std::invalid_argument("Batch size must be greater than zero for custom training.");
        }
        if (!inputs.defined() || !targets.defined()) {
            throw std::invalid_argument("Custom training requires defined inputs and targets.");
        }

        const auto total_samples = inputs.size(0);
        if (total_samples == 0) {
            return;
        }

        auto pinned_inputs = Thot::async_pin_memory(inputs.contiguous());
        auto pinned_targets = Thot::async_pin_memory(targets.contiguous());
        auto host_inputs = pinned_inputs.materialize();
        auto host_targets = pinned_targets.materialize();

        const auto device = model.device();
        const auto batch_size = static_cast<std::int64_t>(options.batch_size);

        auto dice_descriptor = Thot::Loss::Dice();

        model.train();

        for (std::size_t epoch = 0; epoch < options.epochs; ++epoch) {
            double accumulated_loss = 0.0;
            std::size_t batches = 0;

            for (std::int64_t offset = 0; offset < total_samples; offset += batch_size) {
                const auto remaining = total_samples - offset;
                const auto current_batch = std::min<std::int64_t>(batch_size, remaining);
                if (current_batch <= 0) {
                    break;
                }

                auto batch_inputs = host_inputs.narrow(0, offset, current_batch)
                    .to(device, host_inputs.scalar_type(), /*non_blocking=*/true);
                auto batch_targets = host_targets.narrow(0, offset, current_batch)
                    .to(device, host_targets.scalar_type(), /*non_blocking=*/true);

                model.zero_grad();
                auto predictions = model.forward(batch_inputs);
                batch_targets = batch_targets.to(predictions.scalar_type());

                auto dice_loss = Thot::Loss::Details::compute(dice_descriptor, predictions, batch_targets);

                auto clamped_predictions = predictions.clamp(1e-7, 1.0 - 1e-7);
                auto bce_loss = torch::nn::functional::binary_cross_entropy(
                    clamped_predictions,
                    batch_targets,
                    torch::nn::functional::BinaryCrossEntropyFuncOptions().reduction(torch::kMean));

                auto total_loss = dice_loss.mul(options.dice_weight) + bce_loss.mul(options.bce_weight);

                total_loss.backward();
                model.step();

                accumulated_loss += total_loss.detach().item<double>();
                ++batches;
            }

            if (batches > 0)
                accumulated_loss /= static_cast<double>(batches);
            std::cout << "Epoch " << (epoch + 1) << "/" << options.epochs << " - Dice+BCE Loss: " << accumulated_loss << std::endl;
        }
    }
}

int main() {
    Thot::Model model("");
    model.use_cuda(torch::cuda::is_available());

    auto [x1, y1, x2, y2] =
        Thot::Data::Load::Universal("/home/moonfloww/Projects/DATASETS/Image/Satellite/DubaiSegmentationImages/Tile 1/",
            Thot::Data::Type::JPG{"images", {.normalize = true, .pad_to_max_tile=true}},
            Thot::Data::Type::PNG{"masks", {.normalize = true, .pad_to_max_tile=true}}, {.train_fraction = .8f, .test_fraction = .2f, .shuffle = true});

    Thot::Data::Check::Size(x1, "Inputs Raw");
    Thot::Data::Check::Size(y1, "Outputs Raw");

    //797 × 644 pixel

    std::tie(x1, y1) = Thot::Data::Transforms::Augmentation::Flip(x1, y1, {.axes = {"x"}, .frequency = 1.f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transforms::Augmentation::Flip(x1, y1, {.axes = {"y"}, .frequency = 0.5f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transforms::Augmentation::CLAHE(x1, y1, {.frequency = 0.5f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transforms::Augmentation::OpticalDistortion(x1, y1, {.frequency = 0.3f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transforms::Augmentation::AtmosphericDrift(x1, y1, {.frequency = 0.3f, .data_augment = true});
    std::tie(x1, y1) = Thot::Data::Transforms::Augmentation::SunAngleJitter(x1, y1, {.frequency = 0.3f, .data_augment = true});

    Thot::Data::Check::Size(x1, "Inputs Raw");
    Thot::Data::Check::Size(y1, "Targets Raw");


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
    vit_options.attention.dropout = 0.1;
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
    vit_options.patch_embedding.dropout = 0.1;
    vit_options.positional_encoding.type = Thot::Layer::Details::PositionalEncodingType::Learned;
    vit_options.positional_encoding.dropout = 0.1;
    vit_options.residual_dropout = 0.1;
    vit_options.attention_dropout = 0.1;
    vit_options.feed_forward_dropout = 0.1;
    vit_options.pre_norm = true;
    vit_options.final_layer_norm = true;


    const auto patch_size = vit_options.patch_embedding.patch_size;
    const auto tokens_h = (image_height - patch_size) / patch_size + 1;
    const auto tokens_w = (image_width - patch_size) / patch_size + 1;
    const auto patch_area = patch_size * patch_size;
    const auto patch_projection_dim = mask_channels * patch_area;

    model.add(Thot::Block::Transformer::Vision::Encoder(vit_options));
    model.add(Thot::Layer::HardDropout({ .probability = 0.1 }));
    model.add(Thot::Layer::FC({vit_options.embed_dim, patch_projection_dim, true}, Thot::Activation::Identity, Thot::Initialization::XavierUniform));
    model.add(Thot::Layer::PatchUnembed(
{.channels = mask_channels, .tokens_height = tokens_h, .tokens_width = tokens_w, .patch_size = patch_size, .target_height = mask_height, .target_width = mask_width, .align_corners = false },
        Thot::Activation::Sigmoid));

    model.set_loss(Thot::Loss::Dice());
    model.set_optimizer(Thot::Optimizer::AdamW({.learning_rate=1e-4}));


    //Custom loop bcs we use dual-loss
    CustomTrainingOptions training_options{};
    training_options.epochs = 20;
    training_options.batch_size = 8;
    training_options.dice_weight = 0.6;
    training_options.bce_weight = 0.4;

    Train(model, x1, y1, training_options);

    model.evaluate(x2, y2, Thot::Evaluation::Classification, {
        Thot::Metric::Classification::Accuracy,
        Thot::Metric::Classification::Precision,
        Thot::Metric::Classification::Recall,
        Thot::Metric::Classification::JaccardIndexMicro,
        Thot::Metric::Classification::HausdorffDistance,
        Thot::Metric::Classification::BoundaryIoU,
    },
    {.batch_size = 2});


    return 0;
}