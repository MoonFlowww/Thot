#include <cassert>
#include <cmath>
#include <iostream>
#include "layers/layers.hpp"
#include "tensor.hpp"
#include "initializations/initializations.hpp"

int main() {
    using namespace Thot;

    RNNLayer rnn(1, 1, 1, Activation::Tanh, Initialization::Ones);
    rnn.reset_hidden_state();

    Initializers::ones(rnn.W_ih());   // input->hidden weights
    Initializers::ones(rnn.W_hh());   // hidden->hidden weights
    Initializers::zeros(rnn.bias());  // force bias=0 for clean check

    {
        auto wih = rnn.W_ih().download();
        auto whh = rnn.W_hh().download();
        for (float v : wih) assert(std::fabs(v - 1.0f) < 1e-5);
        for (float v : whh) assert(std::fabs(v - 1.0f) < 1e-5);
    }

    // Step 1: input=1
    Utils::Tensor input({1,1});
    input.upload({1.0f});
    auto out1 = rnn.forward(input).download();
    float expected1 = std::tanh(1.0f);   // tanh(1*1 + 0*1 + 0)
    std::cout << "Step1 -> Got: " << out1[0]
          << " Expected: " << expected1 << std::endl;

    assert(std::fabs(out1[0] - expected1) < 1e-4);

    // Step 2: input=0
    input.upload({0.0f});
    auto out2 = rnn.forward(input).download();
    float expected2 = std::tanh(out1[0]); // tanh(0*1 + prev_hidden*1 + 0)
    std::cout << "Step2 -> Got: " << out2[0]
          << " Expected: " << expected2 << std::endl;
    assert(std::fabs(out2[0] - expected2) < 1e-4);


    rnn.reset_hidden_state();
    input.upload({0.0f});
    auto baseline = rnn.forward(input).download();
    float expected_baseline = 0.0f;   // tanh(0*1 + 0*1 + 0)
    std::cout << "Computed out1[0]: " << out1[0] << std::endl;
    std::cout << "Baseline -> Got: " << baseline[0]
          << " Expected: " << expected_baseline << std::endl;
    assert(std::fabs(baseline[0] - expected_baseline) < 1e-4);


    std::cout << "All RNN sequence tests passed." << std::endl;
    return 0;
}
