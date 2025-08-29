#include <cassert>
#include <cmath>
#include <iostream>
#include "activations/activations.hpp"
#include "tensor.hpp"

int main() {
    using namespace Thot;
    Utils::Tensor input({1,3});
    input.upload({-1.0f, 0.0f, 1.0f});
    Utils::Tensor output({1,3});

    // ReLU
    Activations::apply_activation(input, output, Activation::ReLU);
    auto h = output.download();
    assert(h[0] == 0.0f && h[1] == 0.0f && std::fabs(h[2]-1.0f) < 1e-5);

    // Sigmoid
    Activations::apply_activation(input, output, Activation::Sigmoid);
    h = output.download();
    assert(std::fabs(h[0] - (1.0f/(1.0f+std::exp(1.0f)))) < 1e-5);

    // Tanh
    Activations::apply_activation(input, output, Activation::Tanh);
    h = output.download();
    assert(std::fabs(h[2] - std::tanh(1.0f)) < 1e-5);

    // LeakyReLU
    Activations::apply_activation(input, output, Activation::LeakyReLU);
    h = output.download();
    assert(std::fabs(h[0] - (-0.01f)) < 1e-5);

    // ELU
    Activations::apply_activation(input, output, Activation::ELU);
    h = output.download();
    assert(std::fabs(h[0] - (std::exp(-1.0f)-1.0f)) < 1e-5);

    // GELU
    Activations::apply_activation(input, output, Activation::GELU);
    h = output.download();
    float sqrt2_over_pi = 0.7978845608028654f;
    float y = sqrt2_over_pi * (1.0f + 0.044715f * 1.0f * 1.0f * 1.0f);
    float expected_gelu = 0.5f * 1.0f * (1.0f + std::tanh(y));
    assert(std::fabs(h[2] - expected_gelu) < 1e-5);

    // Softmax
    Activations::apply_activation(input, output, Activation::Softmax);
    h = output.download();
    float e1 = std::exp(-1.0f);
    float e2 = std::exp(0.0f);
    float e3 = std::exp(1.0f);
    float sum = e1 + e2 + e3;
    assert(std::fabs(h[0] - e1/sum) < 1e-5);
    assert(std::fabs(h[1] - e2/sum) < 1e-5);
    assert(std::fabs(h[2] - e3/sum) < 1e-5);

    std::cout << "Activation tests passed." << std::endl;
    return 0;
}
