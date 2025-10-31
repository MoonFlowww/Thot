#include "../include/Thot.h"
#include <iostream>
#include <stdexcept>
#include <string>

int main()
{
    Thot::Model model{"adamw_graph_safety"};
    model.add(Thot::Layer::FC({4, 2, true}, Thot::Activation::Identity, Thot::Initialization::Default), "fc");
    model.links({
        Thot::LinkSpec{Thot::Port::parse("@input"), Thot::Port::parse("fc")},
        Thot::LinkSpec{Thot::Port::parse("fc"), Thot::Port::parse("@output")},
    });

    model.set_loss(Thot::Loss::MSE({}));
    model.set_optimizer(Thot::Optimizer::AdamW({.learning_rate = 1e-3}));
    auto inputs = torch::randn({2, 4});
    auto targets = torch::randn({2, 2});
    Thot::TrainOptions options{};
    options.epoch = 1;
    options.batch_size = 1;
    options.graph_mode = Thot::GraphMode::Capture;
    bool threw = false;
    try {
        model.train(inputs, targets, options);
    } catch (const std::runtime_error& error) {
        threw = true;
        const std::string message = error.what();
        if (message.find("AdamW") == std::string::npos) {
            std::cerr << "Expected optimizer name in error message, got: " << message << '\n';
            return 1;
        }
        if (message.find("capture-safe") == std::string::npos) {
            std::cerr << "Expected capture-safe guidance in error message, got: " << message << '\n';
            return 1;
        }
    }
    if (!threw) {
        std::cerr << "Expected graph-mode training with AdamW to throw." << '\n';
        return 1;
    }

    std::cout << "AdamW graph capability test passed." << std::endl;
    return 0;
}
