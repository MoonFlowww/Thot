#include <iostream>

#include <torch/torch.h>
#include "../include/Thot.h"


using DummyConfig = Thot::Core::TrainingConfig<200, 8, true, false>;

int main() {
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

    model.train<DummyConfig>(std::move(dataset));

    model.eval();

    auto test_inputs = torch::tensor({{0.1F, -0.2F}, {0.25F, 0.4F}});
    auto predictions = model.forward(test_inputs);
    auto expected = torch::matmul(test_inputs, weight.unsqueeze(1)) + bias;

    std::cout << "Predictions:\n" << predictions.squeeze() << '\n';
    std::cout << "Expected:\n" << expected.squeeze() << '\n';

    return 0;
}
