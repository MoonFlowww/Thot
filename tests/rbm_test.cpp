#include <cassert>
#include <iostream>
#include "layers/layers.hpp"
#include "tensor.hpp"

int main() {
    using namespace Thot;
    RBMLayer rbm(1, 1, 1, Activation::Sigmoid, Initialization::Ones);

    Utils::Tensor input({1,1});
    input.upload({1.0f});
    auto out = rbm.forward(input).download();
    assert(out.size() == 1);
    assert(out[0] == 0.0f); // hidden states initially zero

    Utils::Tensor grad_out({1,1});
    grad_out.upload({0.0f});
    auto grad_in = rbm.backward(grad_out).download();
    assert(grad_in.size() == 1);

    std::cout << "RBM layer tests passed." << std::endl;
    return 0;
}
