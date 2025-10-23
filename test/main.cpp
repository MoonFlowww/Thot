#include <iostream>
#include <cstddef>
#include <vector>
#include <torch/torch.h>
#include <utility>
#include "../include/Thot.h"


namespace {
    [[nodiscard]] Thot::Block::Details::ResidualOutputOptions make_residual_output_options()
    {
        Thot::Block::Details::ResidualOutputOptions options{};
        options.final_activation = Thot::Activation::GeLU;
        options.dropout = 0.1;
        return options;
    }
}


int main() {


    Thot::Model model;
    model.to_device(torch::cuda::is_available());

    Thot::Layer::FCOptions input_projection_options{32*32, 32*32, true};
    model.add(Thot::Layer::FC(input_projection_options, Thot::Activation::GeLU));

    Thot::Layer::DropoutOptions input_dropout_options{};
    input_dropout_options.probability = 0.1;
    model.add(Thot::Layer::Dropout(input_dropout_options));

    Thot::Block::Transformer::Classic::EncoderOptions encoder_options{};
    encoder_options.layers = 2;
    encoder_options.embed_dim = 64;
    encoder_options.attention.num_heads = 4;
    encoder_options.attention.dropout = 0.1;
    encoder_options.feed_forward.mlp_ratio = 2.0;
    encoder_options.feed_forward.activation = Thot::Activation::GeLU;
    encoder_options.positional_encoding.type = Thot::Block::Transformer::Classic::PositionalEncodingType::Sinusoidal;
    encoder_options.positional_encoding.max_length = static_cast<std::size_t>(sequence_length);
    encoder_options.positional_encoding.dropout = 0.1;
    encoder_options.dropout = 0.05;
    model.add(Thot::Block::Transformer::Classic::Encoder(encoder_options));

    Thot::Layer::FlattenOptions flatten_options{};
    flatten_options.start_dim = 1;
    flatten_options.end_dim = -1;
    model.add(Thot::Layer::Flatten(flatten_options));

    Thot::Layer::FCOptions funnel_options{sequence_length * embed_dim, embed_dim, true};
    model.add(Thot::Layer::FC(funnel_options, Thot::Activation::GeLU));

    std::vector<Thot::Layer::Descriptor> residual_layers{};
    residual_layers.emplace_back(Thot::Layer::FC({embed_dim, embed_dim, true}, Thot::Activation::GeLU));
    residual_layers.emplace_back(Thot::Layer::FC({embed_dim, embed_dim, true}, Thot::Activation::Identity));
    model.add(Thot::Block::Residual(std::move(residual_layers), 2, {}, make_residual_output_options()));

    Thot::Layer::DropoutOptions output_dropout_options{};
    output_dropout_options.probability = 0.1;
    model.add(Thot::Layer::Dropout(output_dropout_options));

    model.add(Thot::Layer::FC({embed_dim, output_dim, true}, Thot::Activation::Identity));

    Thot::Optimizer::AdamWOptions optimizer_options{};
    optimizer_options.learning_rate = 5e-4;
    optimizer_options.weight_decay = 1e-2;
    model.set_optimizer(Thot::Optimizer::AdamW(optimizer_options));

    model.set_loss(Thot::Loss::MSE());

    auto [x1, y1, x2, y2] = Thot::Data::Load::CIFAR10("/home/moonfloww/Projects/DATASETS/CIFAR10", 1.f, 1.f, true);



    auto [fx2, fy2] = Thot::Data::Manipulation::Fraction(x2, y2, 0.05f);

    model.train(x1, y1, {.epoch = 120, .batch_size = 32, .test=std::make_pair(fx2, fy2)});

    model.eval();

    torch::NoGradGuard no_grad;



    auto predictions = model.forward(x2.to(model.device()));

    std::cout << "Predictions:\n" << predictions.cpu().squeeze() << '\n';
    std::cout << "Expected:\n" << y2.squeeze() << '\n';

    return 0;
}
