#include "../../include/Thot.h"

int main() {
    Thot::Model model("");
    model.use_cuda(torch::cuda::is_available());

    model.add(Thot::Layer::Flatten());
    model.add(Thot::Layer::FC({.in_features=3*28*28, .out_features=3*8*8}, Thot::Activation::GeLU, Thot::Initialization::HeUniform));
    model.add(Thot::Layer::FC({.in_features=3*8*8, .out_features=8*8}, Thot::Activation::GeLU, Thot::Initialization::HeUniform));
    model.add(Thot::Layer::FC({.in_features=16, .out_features=10}, Thot::Activation::Identity, Thot::Initialization::HeUniform));

    auto[x1, y1, x2, y2] = Thot::Data::Load::MNIST("/home/moonfloww/Projects/DATASETS/Image/MNIST", 0.1f, 0.1f);
    auto result = Thot::NTK::Compute(model, x1, y1, {.memory_mode = Thot::NTK::MemoryMode::FullGPU, .print_summary = true});

    std::cout << "Kernel shape: " << result.kernel.sizes() << "\n";
    std::cout << "Trace: " << result.stats.trace << "\n";
    std::cout << "Condition number: " << result.stats.condition_number << "\n";

    return 0;
}