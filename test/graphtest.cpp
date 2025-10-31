#include "../include/Thot.h"
#include <iostream>
#include <stdexcept>
#include <string>

int fake_main()
{
    Thot::Model model{"adamw_graph_cpu_regression"};
    model.to_device(torch::cuda::is_available());
    model.add(Thot::Layer::FC({4, 2, true}, Thot::Activation::Identity, Thot::Initialization::Default), "fc");
    model.links({
        Thot::LinkSpec{Thot::Port::parse("@input"), Thot::Port::parse("fc")},
        Thot::LinkSpec{Thot::Port::parse("fc"), Thot::Port::parse("@output")},
    }, /*enable_graph_capture=*/true);

    model.set_loss(Thot::Loss::MSE({}));
    model.set_optimizer(Thot::Optimizer::AdamW({.learning_rate = 1e-3}));

    auto inputs = torch::randn({2, 4});
    auto targets = torch::randn({2, 2});

    Thot::TrainOptions options{};
    options.epoch = 1;
    options.batch_size = 1;
    options.graph_mode = Thot::GraphMode::Capture;

#ifdef TORCH_CUDA_AVAILABLE
    const bool cuda_available = torch::cuda::is_available();
    if (cuda_available) {
        model.to_device(true);
    }
#else
    const bool cuda_available = false;
#endif
    bool capture_threw = false;
    std::string capture_message;
    try {
        model.train(inputs, targets, options);
    } catch (const std::runtime_error& error) {
        capture_threw = true;
        capture_message = error.what();
    }

#ifdef TORCH_CUDA_AVAILABLE
    if (!cuda_available) {
        if (!capture_threw
            || capture_message.find("CUDA graph capture requested but CUDA support is unavailable.") == std::string::npos) {
            std::cerr << "Unexpected behaviour when CUDA unavailable: " << capture_message << '\n';
            return 1;
            }
        std::cout << "CUDA unavailable; regression scenario skipped." << std::endl;
        return 0;
    }
#else
    if (!capture_threw
        || capture_message.find("CUDA graph capture requested but CUDA support is unavailable.") == std::string::npos) {
        std::cerr << "Unexpected behaviour when CUDA unavailable: " << capture_message << '\n';
        return 1;
        }
    std::cout << "CUDA unavailable; regression scenario skipped." << std::endl;
    return 0;
#endif

    if (capture_threw) {
        std::cerr << "Graph capture with AdamW should not throw when CUDA is available: " << capture_message << '\n';
        return 1;
    }

    auto replay_options = options;
    replay_options.graph_mode = Thot::GraphMode::Replay;
    try {
        model.train(inputs, targets, replay_options);
    } catch (const std::exception& error) {
        std::cerr << "Graph replay with AdamW failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "AdamW CUDA graph capture regression test passed." << std::endl;
    return 0;
}