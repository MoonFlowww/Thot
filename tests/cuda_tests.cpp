//
// Created by moonfloww on 29/08/2025.
//
#include <sstream>
#define private public
#define protected public
#include "layers/layers.hpp"
#undef private
#undef protected

#include "activations/activations.hpp"
#include "losses/losses.hpp"
#include "tensor.hpp"
#include "initializations/initializations.hpp"

using namespace Thot;

#define CHECK(cond) do { if(!(cond)) { \
    std::cerr << "[FAIL] " << __func__ << " line " << __LINE__ << std::endl; \
    std::exit(EXIT_FAILURE); \
} } while(0)

void test_rnn() {
    std::cout << "[RUN] test_rnn" << std::endl;
    RNNLayer rnn(1, 1, 1, Activation::Tanh, Initialization::Ones);
    rnn.reset_hidden_state();
    Initializers::ones(rnn.W_ih());
    Initializers::ones(rnn.W_hh());
    Initializers::zeros(rnn.bias());
    auto wih = rnn.W_ih().download();
    auto whh = rnn.W_hh().download();
    for (float v : wih) CHECK(std::fabs(v - 1.0f) < 1e-5);
    for (float v : whh) CHECK(std::fabs(v - 1.0f) < 1e-5);
    Utils::Tensor input({1,1});
    input.upload({1.0f});
    auto out1 = rnn.forward(input).download();
    float expected1 = std::tanh(1.0f);
    std::cout << "Step1 -> Got: " << out1[0] << " Expected: " << expected1 << std::endl;
    CHECK(std::fabs(out1[0] - expected1) < 1e-4);
    input.upload({0.0f});
    auto out2 = rnn.forward(input).download();
    float expected2 = std::tanh(out1[0]);
    std::cout << "Step2 -> Got: " << out2[0] << " Expected: " << expected2 << std::endl;
    CHECK(std::fabs(out2[0] - expected2) < 1e-4);
    rnn.reset_hidden_state();
    input.upload({0.0f});
    auto baseline = rnn.forward(input).download();
    CHECK(std::fabs(baseline[0] - 0.0f) < 1e-4);
    std::cout << "[PASS] test_rnn" << std::endl;
}

void test_fc() {
    std::cout << "[RUN] test_fc" << std::endl;
    FCLayer fc(2, 1, Activation::Linear, Initialization::Ones);
    Initializers::ones(fc.weights_);
    Initializers::zeros(fc.bias_);
    Utils::Tensor input({1, 2});
    input.upload({1.0f, 2.0f});
    auto out = fc.forward(input).download();
    float expected = 3.0f;
    std::cout << "FC forward -> Got: " << out[0] << " Expected: " << expected << std::endl;
    CHECK(std::fabs(out[0] - expected) < 1e-4);
    std::cout << "[PASS] test_fc" << std::endl;
}

void test_conv2d() {
    std::cout << "[RUN] test_conv2d" << std::endl;
    Conv2DLayer conv(1, 2, 2, 1, 1, 1, 0, Activation::Linear, Initialization::Ones);
    Initializers::ones(conv.weights_);
    Initializers::zeros(conv.bias_);
    Utils::Tensor input({1,1,2,2});
    input.upload({1.0f,2.0f,3.0f,4.0f});
    auto out = conv.forward(input).download();
    std::vector<float> expected = {1.0f,2.0f,3.0f,4.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        std::cout << "Conv2D output["<<i<<"]="<< out[i] <<" Expected="<<expected[i]<< std::endl;
        CHECK(std::fabs(out[i]-expected[i]) < 1e-4);
    }
    std::cout << "[PASS] test_conv2d" << std::endl;
}

void test_rbm() {
    std::cout << "[RUN] test_rbm" << std::endl;
    RBMLayer rbm(1,1,1, Activation::Sigmoid, Initialization::Ones);
    Initializers::ones(rbm.weights_);
    Initializers::zeros(rbm.visible_bias_);
    Initializers::zeros(rbm.hidden_bias_);
    Utils::Tensor input({1,1});
    input.upload({1.0f});
    auto out = rbm.forward(input).download();
    auto hidden_act = rbm.hidden_probs_.download();
    std::cout << "RBM hidden activation: " << hidden_act[0] << " output state: " << out[0] << std::endl;
    CHECK(std::fabs(hidden_act[0] - 1.0f) < 1e-4);
    CHECK(std::fabs(out[0]) < 1e-4);
    std::cout << "[PASS] test_rbm" << std::endl;
}

void test_activations() {
    std::cout << "[RUN] test_activations" << std::endl;
    Utils::Tensor input({1,3});
    input.upload({-1.0f, 0.0f, 1.0f});
    auto test = [&](Activation act, const std::vector<float>& expected) {
        Utils::Tensor output({1,3});
        Activations::apply_activation(input, output, act);
        auto host = output.download();
        for (int i = 0; i < 3; ++i) {
            std::cout << static_cast<int>(act) << " output["<<i<<"]="<<host[i] << " expected=" << expected[i] << std::endl;
            CHECK(std::fabs(host[i] - expected[i]) < 1e-4);
        }
    };
    test(Activation::ReLU, {0.0f, 0.0f, 1.0f});
    std::vector<float> sig = {1.0f/(1.0f+std::exp(1.0f)), 0.5f, 1.0f/(1.0f+std::exp(-1.0f))};
    test(Activation::Sigmoid, sig);
    std::vector<float> th = {std::tanh(-1.0f), 0.0f, std::tanh(1.0f)};
    test(Activation::Tanh, th);
    float slope = 0.01f;
    test(Activation::LeakyReLU, {-1.0f*slope, 0.0f, 1.0f});
    std::vector<float> elu = {std::exp(-1.0f)-1.0f, 0.0f, 1.0f};
    test(Activation::ELU, elu);
    float s2pi = 0.7978845608f;
    auto gelu_fn = [&](float x){
        float y = s2pi * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1.0f + std::tanh(y));
    };
    std::vector<float> gelu = {gelu_fn(-1.0f), gelu_fn(0.0f), gelu_fn(1.0f)};
    test(Activation::GELU, gelu);
    float e1 = std::exp(-1.0f);
    float e0 = 1.0f;
    float e2 = std::exp(1.0f);
    float sum = e1 + e0 + e2;
    std::vector<float> softmax = {e1/sum, e0/sum, e2/sum};
    test(Activation::Softmax, softmax);
    std::cout << "[PASS] test_activations" << std::endl;
}

void test_losses() {
    std::cout << "[RUN] test_losses" << std::endl;
    {
        Losses loss(Loss::MSE);
        Utils::Tensor pred({1}); pred.upload({1.0f});
        Utils::Tensor targ({1}); targ.upload({2.0f});
        float result = loss.compute(pred, targ);
        float expected = 0.5f;
        std::cout << "MSE -> " << result << " expected " << expected << std::endl;
        CHECK(std::fabs(result - expected) < 1e-4);
    }
    {
        Losses loss(Loss::MAE);
        Utils::Tensor pred({1}); pred.upload({1.0f});
        Utils::Tensor targ({1}); targ.upload({2.5f});
        float result = loss.compute(pred, targ);
        float expected = 1.5f;
        std::cout << "MAE -> " << result << " expected " << expected << std::endl;
        CHECK(std::fabs(result - expected) < 1e-4);
    }
    {
        Losses loss(Loss::BinaryCrossEntropy);
        Utils::Tensor pred({1}); pred.upload({0.8f});
        Utils::Tensor targ({1}); targ.upload({1.0f});
        float result = loss.compute(pred, targ);
        float expected = -std::log(0.8f);
        std::cout << "BCE -> " << result << " expected " << expected << std::endl;
        CHECK(std::fabs(result - expected) < 1e-4);
    }
    {
        Losses loss(Loss::CategoricalCrossEntropy);
        Utils::Tensor pred({1,2}); pred.upload({0.1f,0.9f});
        Utils::Tensor targ({1,2}); targ.upload({0.0f,1.0f});
        float result = loss.compute(pred, targ);
        float expected = -std::log(0.9f);
        std::cout << "CCE -> " << result << " expected " << expected << std::endl;
        CHECK(std::fabs(result - expected) < 1e-4);
    }
    {
        Losses loss(Loss::SparseCategoricalCrossEntropy);
        Utils::Tensor pred({1,2}); pred.upload({0.1f,0.9f});
        Utils::Tensor targ({1}); targ.upload({1.0f});
        float result = loss.compute(pred, targ);
        float expected = -std::log(0.9f);
        std::cout << "SCCE -> " << result << " expected " << expected << std::endl;
        CHECK(std::fabs(result - expected) < 1e-4);
    }
    {
        Losses loss(Loss::Hinge);
        Utils::Tensor pred({1}); pred.upload({0.6f});
        Utils::Tensor targ({1}); targ.upload({1.0f});
        float result = loss.compute(pred, targ);
        float expected = std::max(0.0f, 1.0f - 0.6f * 1.0f);
        std::cout << "Hinge -> " << result << " expected " << expected << std::endl;
        CHECK(std::fabs(result - expected) < 1e-4);
    }
    {
        Losses loss(Loss::Huber, 1e-8f, 1.0f);
        Utils::Tensor pred({1}); pred.upload({3.0f});
        Utils::Tensor targ({1}); targ.upload({1.0f});
        float result = loss.compute(pred, targ);
        float expected = 1.5f;
        std::cout << "Huber -> " << result << " expected " << expected << std::endl;
        CHECK(std::fabs(result - expected) < 1e-4);
    }
    {
        Losses loss(Loss::KLDivergence);
        Utils::Tensor pred({1}); pred.upload({0.4f});
        Utils::Tensor targ({1}); targ.upload({0.5f});
        float result = loss.compute(pred, targ);
        float expected = 0.5f * std::log(0.5f/0.4f);
        std::cout << "KL -> " << result << " expected " << expected << std::endl;
        CHECK(std::fabs(result - expected) < 1e-4);
    }
    std::cout << "[PASS] test_losses" << std::endl;
}

int main() {
    test_rnn();
    test_fc();
    test_conv2d();
    test_rbm();
    test_activations();
    test_losses();
    std::cout << "All CUDA backend tests passed." << std::endl;
    return 0;
}
