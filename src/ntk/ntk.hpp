#ifndef THOT_NTK_HPP
#define THOT_NTK_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace Thot::NTK {
    enum class KernelType { NTK, NNGP };
    enum class OutputMode { SumOutputs, PerOutput, DiagonalAverage };
    enum class Approximation { Exact, RandomProjection, Nystrom };
    enum class MemoryMode { FullGPU, StreamCPU, OperatorCG, OperatorLanczos };
    enum class Sampling { FullDataset, Subsampled };

    using Device = torch::DeviceType;
    using DType = c10::ScalarType;

    struct Options {
        KernelType   kernel_type      = KernelType::NTK;
        OutputMode   output_mode      = OutputMode::SumOutputs;
        Approximation approximation   = Approximation::Exact;

        std::size_t  max_samples      = 512;
        MemoryMode   memory_mode      = MemoryMode::FullGPU;

        bool         center_kernel    = false;
        bool         normalize_diag   = false;

        bool         compute_eigs     = true;
        int          top_k_eigs       = 20;
        double       ridge_lambda     = 1e-4;

        bool         estimate_lr_bound = true;
        bool         run_kernel_regression = false;

        std::ostream* stream{&std::cout};
        bool print_summary{true};
    };

    // TODO: Create pipelines for NTKStats inside NTK/details
    struct NTKStats {
        // 1. Basic info
        int64_t n_samples{0};
        int64_t n_params_effective{0};
        KernelType kernel_type{KernelType::NTK};
        OutputMode output_mode{OutputMode::SumOutputs};
        bool centered{false};
        bool normalized_diag{false};
        Sampling sampling{Sampling::FullDataset};
        std::vector<int64_t> sample_indices{};

        // 2. Spectrum
        int64_t rank_estimate{0};
        double trace{0.0};
        double frobenius_norm{0.0};
        double lambda_max{0.0};
        double lambda_min_positive{0.0};
        double condition_number{0.0};
        std::vector<double> top_eigenvalues{};
        std::vector<double> eigenvalue_percentiles{};
        double effective_rank{0.0};
        double stable_rank{0.0};

        // 3. Geometry
        double diag_mean{0.0}, diag_std{0.0};
        double offdiag_mean{0.0}, offdiag_std{0.0};
        double within_class_mean{std::numeric_limits<double>::quiet_NaN()}, within_class_std{std::numeric_limits<double>::quiet_NaN()};
        double between_class_mean{std::numeric_limits<double>::quiet_NaN()}, between_class_std{std::numeric_limits<double>::quiet_NaN()};
        double class_separation_score{std::numeric_limits<double>::quiet_NaN()};
        std::vector<double> per_class_self_sim{};

        // 4. LR / dynamics
        double lr_crit{0.0};
        double lr_safe{0.0};
        std::pair<double,double> lr_recommended_range{};
        double spectral_gap{0.0};

        // 5. KRR performance
        bool   krr_ran{false};
        double ridge_lambda_used{0.0};
        double train_loss_krr{std::numeric_limits<double>::quiet_NaN()}, val_loss_krr{std::numeric_limits<double>::quiet_NaN()};
        double train_accuracy_krr{std::numeric_limits<double>::quiet_NaN()}, val_accuracy_krr{std::numeric_limits<double>::quiet_NaN()};
        std::vector<double> per_class_val_accuracy_krr{};
        double label_alignment{std::numeric_limits<double>::quiet_NaN()};

        // 6. Numerical / impl
        Device kernel_device{torch::kCPU};
        DType  kernel_dtype{torch::kFloat32};
        Approximation approximation_mode{Approximation::Exact};
        int64_t feature_dim_used{0};
        bool    is_psd_checked{false};
        double  symmetry_error_max{0.0};
        double  max_abs_entry{0.0};
        int64_t cg_iterations{0};

        // 7. Meta
        Options options_used{};
        std::string  model_name{};
        uint64_t     random_seed{0};
    };

    struct Result {
        torch::Tensor kernel{};
        NTKStats stats{};
    };

    namespace detail {
        inline torch::Tensor flatten_grads(const std::vector<torch::Tensor>& grads) {
            if (grads.empty()) {
                return torch::zeros({0}, torch::TensorOptions().dtype(torch::kFloat32));
            }
            std::vector<torch::Tensor> pieces;
            pieces.reserve(grads.size());
            for (const auto& g : grads) {
                if (!g.defined()) {
                    continue;
                }
                pieces.push_back(g.contiguous().view({-1}));
            }
            if (pieces.empty()) {
                return torch::zeros({0}, torch::TensorOptions().dtype(torch::kFloat32));
            }
            return torch::cat(pieces);
        }

        inline double compute_std(const torch::Tensor& values) {
            if (values.numel() <= 1) {
                return 0.0;
            }
            auto mean = values.mean();
            auto var = (values - mean).pow(2).mean();
            return var.item<double>() > 0.0 ? std::sqrt(var.item<double>()) : 0.0;
        }

        inline double percentile(std::vector<double> values, double p) {
            if (values.empty()) {
                return 0.0;
            }
            std::sort(values.begin(), values.end());
            const double pos = p * static_cast<double>(values.size() - 1);
            const auto idx = static_cast<std::size_t>(pos);
            const double frac = pos - static_cast<double>(idx);
            if (idx + 1 < values.size()) {
                return values[idx] * (1.0 - frac) + values[idx + 1] * frac;
            }
            return values.back();
        }

        inline void maybe_print(std::ostream* stream, bool enabled, const NTKStats& stats) {
            if (!enabled || stream == nullptr) {
                return;
            }
            (*stream) << "[NTK] Samples: " << stats.n_samples
                      << ", Params: " << stats.n_params_effective
                      << ", Trace: " << stats.trace
                      << ", Frobenius: " << stats.frobenius_norm
                      << ", lambda_max: " << stats.lambda_max
                      << ", condition: " << stats.condition_number
                      << "\n";
        }
    }

    template <class Model>
    [[nodiscard]] inline auto Compute(Model& model, torch::Tensor inputs, torch::Tensor targets,
                                      Options options = Options{}) -> Result {
        Result result{};
        NTKStats stats{};

        const auto total_samples = inputs.dim() > 0 ? inputs.size(0) : 0;
        const auto num_samples = std::min<int64_t>(options.max_samples, total_samples);
        if (inputs.dim() == 0 || num_samples == 0) {
            stats.options_used = options;
            result.stats = stats;
            return result;
        }

        inputs = inputs.narrow(0, 0, num_samples);
        if (targets.defined() && targets.size(0) >= num_samples) {
            targets = targets.narrow(0, 0, num_samples);
        }

        stats.n_samples = num_samples;
        stats.kernel_type = options.kernel_type;
        stats.output_mode = options.output_mode;
        stats.centered = options.center_kernel;
        stats.normalized_diag = options.normalize_diag;
        stats.sampling = (num_samples == total_samples) ? Sampling::FullDataset : Sampling::Subsampled;
        stats.sample_indices.reserve(num_samples);
        for (int64_t i = 0; i < num_samples; ++i) {
            stats.sample_indices.push_back(i);
        }

        auto& module_ref = [&]() -> torch::nn::Module& {
            if constexpr (std::is_pointer_v<std::decay_t<Model>>) {
                return *model;
            } else {
                return model;
            }
        }();

        std::vector<torch::Tensor> params;
        for (const auto& p : module_ref.parameters()) {
            if (p.requires_grad()) {
                params.push_back(p);
                stats.n_params_effective += p.numel();
            }
        }

        std::vector<torch::Tensor> flattened_grads;
        flattened_grads.reserve(num_samples);
        for (int64_t i = 0; i < num_samples; ++i) {
            auto sample = inputs[i];
            auto output = module_ref.forward(sample.unsqueeze(0));
            torch::Tensor scalar_output = output;
            if (output.defined() && output.numel() > 1) {
                switch (options.output_mode) {
                    case OutputMode::SumOutputs:
                        scalar_output = output.sum();
                        break;
                    case OutputMode::PerOutput:
                        scalar_output = output.flatten().mean();
                        break;
                    case OutputMode::DiagonalAverage:
                        scalar_output = output.flatten().mean();
                        break;
                }
            }

            auto grads = torch::autograd::grad({scalar_output}, params, {}, /*retain_graph=*/false, /*create_graph=*/false);
            flattened_grads.push_back(detail::flatten_grads(grads).detach());
        }

        auto grad_matrix = torch::stack(flattened_grads);

        stats.feature_dim_used = grad_matrix.size(1);
        auto kernel = torch::matmul(grad_matrix, grad_matrix.transpose(0, 1));

        if (options.center_kernel) {
            const auto mean_all = kernel.mean();
            kernel = kernel - kernel.mean(1, true) - kernel.mean(0, true) + mean_all;
        }
        if (options.normalize_diag) {
            auto diag = kernel.diag();
            auto norm = torch::sqrt(torch::outer(diag, diag) + 1e-12);
            kernel = kernel / norm;
        }

        stats.kernel_device = kernel.device().type();
        stats.kernel_dtype = kernel.scalar_type();
        stats.approximation_mode = options.approximation;

        stats.trace = kernel.diag().sum().item<double>();
        stats.frobenius_norm = kernel.norm().item<double>();
        stats.max_abs_entry = kernel.abs().max().item<double>();
        stats.symmetry_error_max = (kernel - kernel.transpose(0, 1)).abs().max().item<double>();

        if (options.compute_eigs) {
            auto kernel_cpu = kernel.to(torch::kCPU, /*non_blocking=*/true).to(torch::kFloat64);
            auto eig = torch::linalg::eigh(kernel_cpu);
            auto eigenvalues = std::get<0>(eig);
            stats.rank_estimate = torch::linalg::matrix_rank(kernel_cpu).item<int64_t>();
            const auto eigen_vec = eigenvalues.contiguous();
            std::vector<double> eigen_list(eigen_vec.numel());
            std::memcpy(eigen_list.data(), eigen_vec.data_ptr<double>(), eigen_vec.numel() * sizeof(double));

            if (!eigen_list.empty()) {
                std::sort(eigen_list.begin(), eigen_list.end(), std::greater<>());
                stats.lambda_max = eigen_list.front();
                for (auto v : eigen_list) {
                    if (v > 1e-12) {
                        stats.lambda_min_positive = v;
                    }
                }
                stats.condition_number = (stats.lambda_min_positive > 0.0) ? stats.lambda_max / stats.lambda_min_positive : std::numeric_limits<double>::infinity();

                const auto max_top = std::min<std::size_t>(options.top_k_eigs, eigen_list.size());
                stats.top_eigenvalues.assign(eigen_list.begin(), eigen_list.begin() + max_top);

                const std::vector<double> percentiles = {0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99};
                stats.eigenvalue_percentiles.reserve(percentiles.size());
                for (auto p : percentiles) {
                    stats.eigenvalue_percentiles.push_back(detail::percentile(eigen_list, p));
                }

                const double fro2 = stats.frobenius_norm * stats.frobenius_norm;
                stats.stable_rank = (stats.lambda_max > 0.0) ? (fro2 / (stats.lambda_max * stats.lambda_max)) : 0.0;

                const double sum_eig = std::accumulate(eigen_list.begin(), eigen_list.end(), 0.0);
                if (sum_eig > 0.0) {
                    double entropy = 0.0;
                    for (auto v : eigen_list) {
                        const double p = v / sum_eig;
                        if (p > 0.0) {
                            entropy -= p * std::log(p);
                        }
                    }
                    stats.effective_rank = std::exp(entropy);
                }

                if (eigen_list.size() > 1) {
                    stats.spectral_gap = eigen_list.front() - eigen_list[1];
                }
            }
            stats.is_psd_checked = true;
        }

        auto diag = kernel.diag();
        stats.diag_mean = diag.mean().item<double>();
        stats.diag_std = detail::compute_std(diag);

        auto mask = torch::ones_like(kernel, torch::TensorOptions().dtype(torch::kBool));
        mask.diagonal().fill_(false);
        auto offdiag = kernel.masked_select(mask);
        stats.offdiag_mean = offdiag.numel() > 0 ? offdiag.mean().item<double>() : 0.0;
        stats.offdiag_std = offdiag.numel() > 0 ? detail::compute_std(offdiag) : 0.0;

        if (targets.defined() && targets.numel() == num_samples) {
            auto labels = targets.flatten().to(torch::kCPU, /*non_blocking=*/true);
            std::unordered_map<int64_t, std::vector<int64_t>> by_class;
            for (int64_t i = 0; i < num_samples; ++i) {
                by_class[labels[i].item<int64_t>()].push_back(i);
            }

            std::vector<double> within_values;
            std::vector<double> between_values;
            stats.per_class_self_sim.reserve(by_class.size());

            for (const auto& [label, idxs] : by_class) {
                if (idxs.empty()) {
                    continue;
                }
                auto idx_tensor = torch::tensor(idxs, torch::TensorOptions().dtype(torch::kLong));
                auto submatrix = kernel.index({idx_tensor, idx_tensor});
                const double mean_sub = submatrix.mean().item<double>();
                stats.per_class_self_sim.push_back(mean_sub);
                within_values.push_back(mean_sub);
            }

            std::vector<int64_t> all_indices(num_samples);
            std::iota(all_indices.begin(), all_indices.end(), 0);
            auto all_idx_tensor = torch::tensor(all_indices, torch::TensorOptions().dtype(torch::kLong));
            auto full_matrix = kernel.index({all_idx_tensor, all_idx_tensor});
            auto between_mask = torch::zeros_like(full_matrix, torch::TensorOptions().dtype(torch::kBool));

            for (const auto& [label_i, idxs_i] : by_class) {
                for (const auto& [label_j, idxs_j] : by_class) {
                    if (label_i == label_j) {
                        continue;
                    }
                    auto idx_i = torch::tensor(idxs_i, torch::TensorOptions().dtype(torch::kLong));
                    auto idx_j = torch::tensor(idxs_j, torch::TensorOptions().dtype(torch::kLong));
                    between_mask.index_put_({idx_i.unsqueeze(1), idx_j.unsqueeze(0)}, true);
                }
            }

            auto between = full_matrix.masked_select(between_mask);
            if (between.numel() > 0) {
                stats.between_class_mean = between.mean().item<double>();
                stats.between_class_std = detail::compute_std(between);
                between_values.push_back(stats.between_class_mean);
            }
            if (!within_values.empty()) {
                stats.within_class_mean = std::accumulate(within_values.begin(), within_values.end(), 0.0) / static_cast<double>(within_values.size());
                stats.within_class_std = detail::compute_std(torch::tensor(within_values, torch::TensorOptions().dtype(torch::kFloat64)));
            }
            if (!between_values.empty() && !within_values.empty()) {
                stats.class_separation_score = stats.between_class_mean - stats.within_class_mean;
            }
        }

        if (options.estimate_lr_bound && stats.lambda_max > 0.0) {
            stats.lr_crit = 2.0 / (stats.lambda_max + options.ridge_lambda);
            stats.lr_safe = stats.lr_crit * 0.5;
            stats.lr_recommended_range = {stats.lr_safe * 0.5, stats.lr_safe};
        }

        if (options.run_kernel_regression && targets.defined()) {
            auto eye = torch::eye(num_samples, kernel.options());
            auto regularized = kernel + options.ridge_lambda * eye;
            auto alpha = torch::linalg_solve(regularized, targets); // shape [n, ...]
            auto preds = torch::matmul(kernel, alpha);
            auto diff = preds - targets;
            stats.train_loss_krr = diff.pow(2).mean().item<double>();

            if (targets.scalar_type() == torch::kLong || targets.scalar_type() == torch::kInt) {
                auto predicted_labels = preds.argmax(-1);
                stats.train_accuracy_krr = (predicted_labels == targets).to(torch::kFloat32).mean().item<double>();
            }
            stats.ridge_lambda_used = options.ridge_lambda;
            stats.krr_ran = true;
        }

        stats.options_used = options;
        stats.model_name = typeid(Model).name();
        stats.random_seed = static_cast<uint64_t>(std::random_device{}());

        detail::maybe_print(options.stream, options.print_summary, stats);

        result.kernel = kernel;
        result.stats = stats;
        return result;
    }
}

#endif // THOT_NTK_HPP