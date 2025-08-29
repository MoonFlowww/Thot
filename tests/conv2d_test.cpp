#include <cassert>
#include <cmath>
#include <iostream>
#include "layers/layers.hpp"
#include "tensor.hpp"

int main() {
    using namespace Thot;
    Conv2DLayer conv(1, 2, 2, 1, 1, 1, 0, Activation::Linear, Initialization::Ones);

    Utils::Tensor input({1,1,2,2});
    input.upload({1.0f, 2.0f, 3.0f, 4.0f});

    auto out = conv.forward(input).download();
    assert(out.size() == 4);
    assert(std::fabs(out[0] - 1.0f) < 1e-4);
    assert(std::fabs(out[1] - 2.0f) < 1e-4);
    assert(std::fabs(out[2] - 3.0f) < 1e-4);
    assert(std::fabs(out[3] - 4.0f) < 1e-4);

    std::cout << "Conv2D layer tests passed." << std::endl;
    return 0;
}
