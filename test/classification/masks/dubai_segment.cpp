#include "../../../include/Thot.h"

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

    model.set_loss(Thot::Loss::MSE());
    model.set_optimizer(Thot::Optimizer::AdamW({.learning_rate=1e-4}));
    model.train(x1, y1, {.epoch = 20, .batch_size=8, .shuffle = true, .restore_best_state=true, .enable_amp = true});

    return 0;
}