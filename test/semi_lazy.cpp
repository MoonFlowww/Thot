#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include <torch/torch.h>

#include "../include/Thot.h"

namespace {
    void homemade_train(
        Thot::Model& model,
        Thot::Core::SupervisedDataset dataset,
        std::size_t epochs = Thot::Core::DefaultTrainingConfig::epochs,
        std::size_t batch_size = Thot::Core::DefaultTrainingConfig::batch_size
    ) {
        if (!model.has_optimizer()) {
            throw std::logic_error("homemade_train requires the model to have an optimizer set.");
        }
        if (!model.has_loss()) {
            throw std::logic_error("homemade_train requires the model to have a loss function set.");
        }
        if (dataset.empty()) {
            return;
        }

        auto device = torch::Device(torch::kCPU);
        model.train();
        model.to(device);

        auto rng = std::mt19937{std::random_device{}()};

        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            std::shuffle(dataset.begin(), dataset.end(), rng);

            for (std::size_t offset = 0; offset < dataset.size(); offset += batch_size) {
                const auto batch_end = std::min(offset + batch_size, dataset.size());
                if (batch_end <= offset) {
                    continue;
                }

                std::vector<torch::Tensor> inputs;
                std::vector<torch::Tensor> targets;
                inputs.reserve(batch_end - offset);
                targets.reserve(batch_end - offset);

                for (std::size_t index = offset; index < batch_end; ++index) {
                    inputs.push_back(dataset[index].first.to(device));
                    targets.push_back(dataset[index].second.to(device));
                }

                const auto batch_inputs = torch::stack(inputs);
                const auto batch_targets = torch::stack(targets);

                const auto predictions = model.forward(batch_inputs);
                auto loss = model.compute_loss(predictions, batch_targets);

                model.zero_grad();
                loss.backward();
                model.step();
            }
        }
    }
}

int instance() {
    torch::manual_seed(42);

    Thot::Model model;
    model.add(Thot::Layer::FC({2, 8}, Thot::Activation::ReLU));
    model.add(Thot::Layer::FC({8, 1}, Thot::Activation::Identity));

    model.set_loss(Thot::Loss::MSE());
    model.set_optimizer(Thot::Optimizer::SGD({.learning_rate = 0.1}));

    Thot::Core::SupervisedDataset dataset;
    dataset.reserve(32);

    const auto weight = torch::tensor({2.0F, -3.0F});
    const auto bias = torch::tensor({0.5F});

    for (int64_t i = 0; i < 32; ++i) {
        auto input = torch::randn({2});
        auto target = torch::dot(input, weight) + bias;
        dataset.emplace_back(input, target.unsqueeze(0));
    }

    homemade_train(model, std::move(dataset));

    model.eval();

    auto test_inputs = torch::tensor({{0.1F, -0.2F}, {0.25F, 0.4F}});
    auto predictions = model.forward(test_inputs);
    auto expected = torch::matmul(test_inputs, weight.unsqueeze(1)) + bias;

    std::cout << "Predictions:\n" << predictions.squeeze() << '\n';
    std::cout << "Expected:\n" << expected.squeeze() << '\n';

    return 0;
}