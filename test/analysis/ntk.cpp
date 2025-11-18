#include "../../include/Thot.h"



int main() {
    Thot::Model model("");
    model.use_cuda(torch::cuda::is_available());

    model.add(Thot::Layer::Flatten());
    model.add(Thot::Layer::FC({.in_features=3*28*28, .out_features=3*8*8}, Thot::Activation::GeLU, Thot::Initialization::HeUniform));
    model.add(Thot::Layer::FC({.in_features=3*8*8, .out_features=8*8}, Thot::Activation::GeLU, Thot::Initialization::HeUniform));
    model.add(Thot::Layer::FC({.in_features=16, .out_features=10}, Thot::Activation::Identity, Thot::Initialization::HeUniform));

    auto[x1, y1, x2, y2] = Thot::Data::Load::MNIST("/home/moonfloww/Projects/DATASETS/Image/MNIST", 0.1f, 0.1f);
    auto stage_for_device = [&](torch::Tensor tensor) {
        const auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        auto pinned = Thot::async_pin_memory(std::move(tensor));
        auto host_tensor = pinned.materialize();
        const bool non_blocking = torch::cuda::is_available() && host_tensor.is_pinned();
        return host_tensor.to(device, host_tensor.scalar_type(), non_blocking);
    };

    x1 = stage_for_device(x1);
    y1 = stage_for_device(y1);

    auto result = Thot::NTK::Compute(model, x1, y1,
        {.kernel_type = Thot::NTK::KernelType::NTK,
        .output_mode = Thot::NTK::OutputMode::SumOutputs,
        .approximation = Thot::NTK::Approximation::Exact,
        .max_samples = 128,
        .memory_mode = Thot::NTK::MemoryMode::FullGPU,
        .subsample_seed = 42,
        .random_projection_dim = 256,
        .nystrom_rank = 256,
        .operator_iterations = 32,
        .center_kernel = false,
        .normalize_diag = false,
        .compute_eigs = true,
        .top_k_eigs = 20,
        .ridge_lambda = 1e-4,
        .estimate_lr_bound = true,
        .run_kernel_regression = false,
        .stream = &std::cout,
        .print_summary = true});


    std::cout << "Kernel shape: " << result.kernel.sizes() << "\n";
    std::cout << "Trace: " << result.stats.trace << "\n";
    std::cout << "Condition number: " << result.stats.condition_number << "\n";

    return 0;
}