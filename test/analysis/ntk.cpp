#include "../../include/Thot.h"

int main() {
    Thot::Model model("");
    model.use_cuda(torch::cuda::is_available());

    model.add(Thot::Layer::Flatten());
    model.add(Thot::Layer::FC({.in_features=3*28*28, .out_features=3*8*8}, Thot::Activation::GeLU, Thot::Initialization::HeUniform));
    model.add(Thot::Layer::FC({.in_features=3*8*8, .out_features=8*8}, Thot::Activation::GeLU, Thot::Initialization::HeUniform));
    model.add(Thot::Layer::FC({.in_features=16, .out_features=10}, Thot::Activation::Identity, Thot::Initialization::HeUniform));

    auto[x1, y1, x2, y2] = Thot::Data::Load::MNIST("/home/moonfloww/Projects/DATASETS/Image/MNIST", 0.1f, 0.1f);


    return 0;
}