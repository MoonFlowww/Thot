#include <cassert>
#include <cmath>
#include <iostream>
#include "layers/layers.hpp"
#include "tensor.hpp"
#include "initializations/initializations.hpp"

int main() {
    using namespace Thot;
    FCLayer fc(1, 1, Activation::Linear, Initialization::Ones);

    Utils::Tensor input({1,1});
    input.upload({2.0f});

    auto out = fc.forward(input).download();
    assert(std::fabs(out[0] - 2.0f) < 1e-4);

    Utils::Tensor grad_out({1,1});
    grad_out.upload({1.0f});
    auto grad_in = fc.backward(grad_out).download();
    assert(std::fabs(grad_in[0] - 1.0f) < 1e-4);

    std::cout << "FC layer tests passed." << std::endl;
    return 0;
}
