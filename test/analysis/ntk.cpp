#include "../../include/Thot.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

namespace {
    using namespace torch::indexing;

    struct ToyDataset {
        torch::Tensor inputs;  // [N, 1, 2]
        torch::Tensor labels;  // [N]
    };

    ToyDataset make_spiral_dataset(int64_t samples_per_class, int64_t num_classes, double noise_std = 0.2) {
        const int64_t total_samples = samples_per_class * num_classes;
        auto input_opts = torch::TensorOptions().dtype(torch::kFloat32);
        auto label_opts = torch::TensorOptions().dtype(torch::kLong);

        auto inputs = torch::zeros({total_samples, 2}, input_opts);
        auto labels = torch::zeros({total_samples}, label_opts);

        for (int64_t class_idx = 0; class_idx < num_classes; ++class_idx) {
            const auto start = class_idx * samples_per_class;
            const auto end = start + samples_per_class;
            auto slice = Slice(start, end);

            auto r = torch::linspace(0.0, 1.0, samples_per_class, input_opts);
            auto theta = torch::linspace(class_idx * 4.0, (class_idx + 1) * 4.0, samples_per_class, input_opts);
            theta += torch::randn({samples_per_class}, input_opts) * noise_std;

            inputs.index_put_({slice, 0}, r * torch::sin(theta));
            inputs.index_put_({slice, 1}, r * torch::cos(theta));
            labels.index_put_({slice}, torch::full({samples_per_class}, class_idx, label_opts));
        }

        inputs = inputs.view({total_samples, 1, 2});
        return {inputs, labels};
    }

    void print_vector(const std::vector<double>& values, const std::string& label, std::size_t max_items = 10) {
        std::cout << label;
        if (values.empty()) {
            std::cout << " <empty>\n";
            return;
        }

        const auto limit = std::min(values.size(), max_items);
        std::cout << " [";
        for (std::size_t i = 0; i < limit; ++i) {
            std::cout << std::fixed << std::setprecision(4) << values[i];
            if (i + 1 < limit) {
                std::cout << ", ";
            }
        }
        if (limit < values.size()) {
            std::cout << ", ...";
        }
        std::cout << "]\n";
    }
}

int main() {
    Thot::Model model("");
    const auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    model.use_cuda(torch::cuda::is_available());

    const int64_t hidden_dim1 = 64;
    const int64_t hidden_dim2 = 64;
    const int64_t num_classes = 3;
    model.add(Thot::Layer::Flatten());
    model.add(Thot::Layer::FC({.in_features = 2, .out_features = hidden_dim1}, Thot::Activation::GeLU,
                              Thot::Initialization::HeUniform));
    model.add(Thot::Layer::FC({.in_features = hidden_dim1, .out_features = hidden_dim2}, Thot::Activation::GeLU,
                              Thot::Initialization::HeUniform));
    model.add(Thot::Layer::FC({.in_features = hidden_dim2, .out_features = num_classes}, Thot::Activation::Identity,
                              Thot::Initialization::HeUniform));
    model.to(device);

    const int64_t samples_per_class = 64;
    auto [inputs, labels] = make_spiral_dataset(samples_per_class, num_classes);

    inputs = inputs.to(device);
    labels = labels.to(device);

    Thot::NTK::Options options;
    options.kernel_type = Thot::NTK::KernelType::NTK;
    options.output_mode = Thot::NTK::OutputMode::SumOutputs;
    options.approximation = Thot::NTK::Approximation::Exact;
    options.memory_mode = Thot::NTK::MemoryMode::FullGPU;
    options.max_samples = inputs.size(0);
    options.center_kernel = true;
    options.normalize_diag = false;
    options.compute_eigs = true;
    options.top_k_eigs = 12;
    options.ridge_lambda = 1e-3;
    options.estimate_lr_bound = true;
    options.run_kernel_regression = true;// solves KRR with the NTK as the kernel
    options.stream = &std::cout;
    options.print_summary = true;

    auto result = Thot::NTK::Compute(model, inputs, labels, options);

    const auto& stats = result.stats;
    std::cout << "\n[NTK] ===== Detailed statistics =====\n";
    std::cout << "Samples used       : " << stats.n_samples << "\n";
    std::cout << "Effective params   : " << stats.n_params_effective << "\n";
    std::cout << "Trace / Frobenius  : " << stats.trace << " / " << stats.frobenius_norm << "\n";
    std::cout << "Top eigenvalue     : " << stats.lambda_max << "\n";
    std::cout << "Condition number   : " << stats.condition_number << "\n";
    std::cout << "Diag mean / std    : " << stats.diag_mean << " / " << stats.diag_std << "\n";
    std::cout << "Off-diag mean / std: " << stats.offdiag_mean << " / " << stats.offdiag_std << "\n";

    print_vector(stats.top_eigenvalues, "Top eigenvalues");
    print_vector(stats.eigenvalue_percentiles, "Eigenvalue percentiles");

    if (stats.lr_recommended_range.first) {
        std::cout << "Suggested LR range : [" << stats.lr_recommended_range.first << ", "
                  << stats.lr_recommended_range.second << "]\n";
    }

    if (stats.krr_ran) {
        std::cout << "\n[NTK] Kernel regression diagnostics\n";
        std::cout << "Train loss (KRR)   : " << stats.train_loss_krr << "\n";
        if (!std::isnan(stats.train_accuracy_krr)) {
            std::cout << "Train accuracy (KRR): " << stats.train_accuracy_krr * 100.0 << "%\n";
        }
        if (!std::isnan(stats.val_loss_krr)) {
            std::cout << "Val loss (aligned) : " << stats.val_loss_krr << "\n";
        }
        if (!std::isnan(stats.val_accuracy_krr)) {
            std::cout << "Val accuracy (KRR) : " << stats.val_accuracy_krr * 100.0 << "%\n";
        }
    }

    if (result.kernel.defined()) {
        auto kernel_cpu = result.kernel.to(torch::kCPU);
        std::cout << "\n[NTK] Kernel matrix shape: " << kernel_cpu.sizes() << "\n";
    }
    return 0;
}
