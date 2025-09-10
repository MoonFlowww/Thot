//
// Created by moonfloww on 29/08/2025.
//
#include <sstream>
#include <memory>
#include <vector>
#include <cmath>

#define private public
#define protected public
#undef private
#undef protected

#include "activations/activations.hpp"
#include "losses/losses.hpp"
#include "tensor.hpp"
#include "initializations/initializations.hpp"
#include "attentions/attentions.hpp"
#include "layers/layers.hpp"
#include "optimizations/optimizations.hpp"
#include "network.hpp"
#include "layernorm/layernorm.hpp"

using namespace Thot;

#define CHECK(cond) do { if(!(cond)) { \
    std::cerr << "[FAIL] " << __func__ << " line " << __LINE__ << std::endl; \
    std::exit(EXIT_FAILURE); \
} } while(0)



void test_rnn() {
    std::cout << "[RUN] test_rnn" << std::endl;
    RNNLayer rnn(1, 1, 1, Activation::Tanh, Initialization::Ones);
    rnn.reset_hidden_state();
    Initializations::ones(rnn.W_ih());
    Initializations::ones(rnn.W_hh());
    Initializations::zeros(rnn.bias());
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
    Initializations::ones(fc.weights());
    Initializations::zeros(fc.bias());
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
    Initializations::ones(conv.weights());
    Initializations::zeros(conv.bias());
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
    Initializations::ones(rbm.weights());
    Initializations::zeros(rbm.visible_bias());
    Initializations::zeros(rbm.hidden_bias());
    Utils::Tensor input({1,1});
    input.upload({1.0f});
    auto out = rbm.forward(input).download();
    auto hidden_act = rbm.hidden_probs().download();
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

void test_mha_dimension_check() {
    std::cout << "[RUN] test_mha_dimension_check" << std::endl;
    bool threw = false;
    try {
        Thot::Attention::MHA(10, 3, Thot::Initialization::Xavier);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    std::cout << "[PASS] test_mha_dimension_check" << std::endl;
}

void test_mha() {
    std::cout << "[RUN] test_mha_forward_backward" << std::endl;
    auto layer = Thot::Attention::MHA(4, 2, Thot::Initialization::Xavier);
    auto mha = std::dynamic_pointer_cast<MHAAttentionLayer>(layer);

    // Set projections: Q and K zero, V and O identity
    Initializations::zeros(mha->W_q()); Initializations::zeros(mha->W_k());
    Initializations::zeros(mha->b_q()); Initializations::zeros(mha->b_k());
    Initializations::zeros(mha->b_v()); Initializations::zeros(mha->b_o());



    std::vector<float> identity(16, 0.0f);
    for (int i = 0; i < 4; ++i) identity[i * 4 + i] = 1.0f;
    mha->W_v().upload(identity);
    mha->W_o().upload(identity);

    Utils::Tensor input({1, 2, 4});
    input.upload(std::vector<float>(8, 0.0f));
    auto out = mha->forward(input).download();
    for (float v : out) {
        std::cout << "MHA output val: " << v << std::endl;
        CHECK(std::fabs(v) < 1e-5);
    }

    Utils::Tensor grad_out({1, 2, 4});
    grad_out.upload(std::vector<float>(8, 1.0f));
    auto grad_in = mha->backward(grad_out).download();
    for (float v : grad_in) {
        std::cout << "MHA grad input: " << v << std::endl;
        CHECK(std::fabs(v - 1.0f) < 1e-4);
    }
    std::cout << "[PASS] test_mha_forward_backward" << std::endl;
    test_mha_dimension_check();
}


void test_mla_4d_input() {
    std::cout << "[RUN] test_mla_4d_input" << std::endl;
    auto layer = Thot::Attention::MLA(3 * 32 * 32, 2, 64, Thot::Initialization::Xavier);
    Utils::Tensor input({1, 3, 32, 32});
    input.upload(std::vector<float>(1 * 3 * 32 * 32, 0.0f));
    bool threw = false;
    try {
        layer->forward(input);
    } catch (...) {
        threw = true;
    }
    CHECK(!threw);
    std::cout << "[PASS] test_mla_4d_input" << std::endl;
}

void test_muon_optimizer() {
    std::cout << "[RUN] test_muon_optimizer" << std::endl;
    auto opt = Optimizer::Muon(0.1f, 0.9f, 0.01f);
    Utils::Tensor w({1});
    w.upload({1.0f});
    Utils::Tensor g({1});
    g.upload({1.0f});
    opt->update(w, g);
    auto host = w.download();
    std::cout << "Muon weight: " << host[0] << std::endl;
    CHECK(host[0] < 1.0f);
    std::cout << "[PASS] test_muon_optimizer" << std::endl;
}


void test_spike_layer() {
    std::cout << "[RUN] test_spike_layer" << std::endl;
    Network net("spike_net");
    net.add(Layer::FC(1,1, Activation::Linear, Initialization::Ones));
    net.add(Layer::Spike(1, 1.0f));
    net.add(Layer::FC(1,1, Activation::Linear, Initialization::Ones));

    Utils::Tensor input({1,1});
    input.upload({1.5f});
    auto out = net.forward_gpu(input).download();
    CHECK(std::fabs(out[0] - 1.0f) < 1e-4);
    std::cout << "[PASS] test_spike_layer" << std::endl;
}

void test_sparse_contractive_ae() {
    std::cout << "[RUN] test_sparse_contractive_ae" << std::endl;
    SparseContractiveAELayer base_layer(2, 2, Activation::Sigmoid, Initialization::Ones, false, false);
    Utils::Tensor input({1,2});
    input.upload({0.5f, 0.25f});
    base_layer.forward(input);
    float base_reg = base_layer.regularization_loss();

    SparseContractiveAELayer reg_layer(2, 2, Activation::Sigmoid, Initialization::Ones, true, true);
    reg_layer.forward(input);
    float reg = reg_layer.regularization_loss();
    std::cout << "base=" << base_reg << " reg=" << reg << std::endl;
    CHECK(reg > base_reg);
    std::cout << "[PASS] test_sparse_contractive_ae" << std::endl;
}


void test_rms_layernorm() {
    std::cout << "[RUN] test_rms_layernorm" << std::endl;
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4];
    LayerNorm::RMSLayerNorm(input, output, 1, 4);
    float mean_sq = (1.0f + 4.0f + 9.0f + 16.0f) / 4.0f;
    float scale = 1.0f / std::sqrt(mean_sq + 1e-5f);
    for (int i = 0; i < 4; ++i) {
        float expected = input[i] * scale;
        CHECK(std::fabs(output[i] - expected) < 1e-4);
    }
    std::cout << "[PASS] test_rms_layernorm" << std::endl;
}

void test_dyt_layernorm() {
    std::cout << "[RUN] test_dyt_layernorm" << std::endl;
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4];
    LayerNorm::DyTLayerNorm(input, output, 1, 4);
    float mean_sq = (1.0f + 4.0f + 9.0f + 16.0f) / 4.0f;
    float scale = 1.0f / std::sqrt(mean_sq + 1e-5f);
    for (int i = 0; i < 4; ++i) {
        float expected = std::tanh(input[i] * scale);
        CHECK(std::fabs(output[i] - expected) < 1e-4);
    }
    std::cout << "[PASS] test_dyt_layernorm" << std::endl;
}

void test_softmax_numerical_stability() {
    std::cout << "[RUN] test_softmax_numerical_stability" << std::endl;
    Utils::Tensor input({1, 3});
    input.upload({1000.0f, 1001.0f, 1002.0f});
    Utils::Tensor output({1, 3});
    Activations::apply_activation(input, output, Activation::Softmax);
    auto host = output.download();

    float maxv = std::max(std::max(1000.0f, 1001.0f), 1002.0f);
    float e0 = std::exp(1000.0f - maxv);
    float e1 = std::exp(1001.0f - maxv);
    float e2 = std::exp(1002.0f - maxv);
    float sum = e0 + e1 + e2;
    std::vector<float> expected = {e0 / sum, e1 / sum, e2 / sum};

    float total = 0.0f;
    for (int i = 0; i < 3; ++i) {
        total += host[i];
        CHECK(!std::isnan(host[i]));
        CHECK(std::fabs(host[i] - expected[i]) < 1e-6);
    }
    CHECK(std::fabs(total - 1.0f) < 1e-6);

    Losses cce(Loss::CategoricalCrossEntropy);
    Utils::Tensor targ({1, 3});
    targ.upload({0.0f, 0.0f, 1.0f});
    float loss = cce.compute(output, targ);
    float expected_loss = -std::log(expected[2]);
    CHECK(std::fabs(loss - expected_loss) < 1e-4);

    Losses scce(Loss::SparseCategoricalCrossEntropy);
    Utils::Tensor targ_s({1});
    targ_s.upload({2.0f});
    float loss_s = scce.compute(output, targ_s);
    CHECK(std::fabs(loss_s - expected_loss) < 1e-4);

    std::cout << "[PASS] test_softmax_numerical_stability" << std::endl;
}


int main() {
    test_rnn();
    test_fc();
    test_conv2d();
    test_rbm();
    test_activations();
    test_losses();
    test_softmax_numerical_stability();
    test_mha();
    test_mla_4d_input();
    test_muon_optimizer();
    test_spike_layer();
    test_sparse_contractive_ae();
    test_rms_layernorm();
    test_dyt_layernorm();
    std::cout << "All CUDA backend tests passed." << std::endl;
    return 0;
}