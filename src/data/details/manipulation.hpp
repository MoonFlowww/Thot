#ifndef THOT_MANIPULATION_HPP
#define THOT_MANIPULATION_HPP
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <torch/torch.h>


#include "manipulation/normalization/ehlers.hpp"
#include "manipulation/normalization/power.hpp"
#include "manipulation/normalization/zscore.hpp"

namespace Thot::Data::Manipulation {
    namespace Details {
        inline bool augmentation_enabled(const std::optional<bool>& data_augment) {
            return !(data_augment.has_value() && !data_augment.value());
        }

        inline double clamp_frequency(const std::optional<double>& frequency) {
            if (!frequency.has_value())
                return 1.0;
            return std::clamp(frequency.value(), 0.0, 1.0);
        }


        inline torch::Tensor select_indices_by_frequency(int64_t batch_size, const std::optional<double>& frequency,
                                                         const torch::Device& device) {
            const auto long_options = torch::TensorOptions().dtype(torch::kLong).device(device);
            if (batch_size <= 0) {
                return torch::empty({0}, long_options);
            }
            const double f = clamp_frequency(frequency);
            if (f <= 0.0) {
                return torch::empty({0}, long_options);
            }
            if (f >= 1.0) {
                return torch::arange(batch_size, long_options);
            }
            int64_t stride = static_cast<int64_t>(std::round(1.0 / f));
            stride = std::max<int64_t>(1, stride);
            return torch::arange(0,batch_size,stride, long_options);
        }

        inline int64_t axis_token_to_dim(const std::string& token, int64_t tensor_dim) {
            if (token.empty()) {
                throw std::invalid_argument("Flip axis tokens must not be empty.");
            }

            try {
                std::size_t processed = 0;
                const auto value = std::stoll(token, &processed, 10);
                if (processed == token.size()) {
                    auto normalized = value;
                    if (normalized < 0) {
                        normalized += tensor_dim;
                    }
                    if (normalized < 0 || normalized >= tensor_dim) {
                        throw std::out_of_range("Flip axis index out of range.");
                    }
                    return normalized;
                }
            } catch (const std::invalid_argument&) {
                // Continue below for named axes.
            } catch (const std::out_of_range&) {
                throw std::out_of_range("Flip axis index out of range.");
            }

            std::string lowered = token;
            std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });

            int64_t offset = 0;
            if (lowered == "x") {
                offset = -1;
            } else if (lowered == "y") {
                offset = -2;
            } else if (lowered == "z") {
                offset = -3;
            } else {
                throw std::invalid_argument("Unsupported flip axis token: " + token);
            }

            const auto dim_index = tensor_dim + offset;
            if (dim_index < 0 || dim_index >= tensor_dim) {
                throw std::out_of_range("Flip axis token is incompatible with tensor rank.");
            }

            return dim_index;
        }

        inline std::vector<int64_t> parse_flip_axes(const std::vector<std::string>& axes, int64_t tensor_dim) {
            std::vector<int64_t> dims;
            dims.reserve(axes.size());
            for (const auto& axis : axes) {
                dims.push_back(axis_token_to_dim(axis, tensor_dim));
            }
            return dims;
        }
    }

    inline std::pair<torch::Tensor, torch::Tensor> Flip(const torch::Tensor& tensor, const torch::Tensor& target, const std::vector<std::string>& axes,
                                                        std::optional<double> frequency = 0.3, std::optional<bool> data_augment = true, bool show_progress = true) {
        [[maybe_unused]] const bool show = show_progress;
        if (!Details::augmentation_enabled(data_augment)) {
            return {tensor, target};
        }
        if (!tensor.defined() || !target.defined() || tensor.dim() == 0 || target.dim() == 0) {
            return {tensor, target};
        }
        if (tensor.size(0) != target.size(0)) {
            throw std::invalid_argument("Inputs and targets must have matching batch dimensions for Flip augmentation.");
        }
        if (axes.empty()) {
            return {tensor, target};
        }

        const auto dims = Details::parse_flip_axes(axes, tensor.dim());
        if (dims.empty()) {
            return {tensor, target};
        }

        auto selected_indices = Details::select_indices_by_frequency(tensor.size(0), frequency, tensor.device());
        if (selected_indices.numel() == 0) {
            return {tensor, target};
        }

        auto selected_inputs = tensor.index_select(0, selected_indices).clone();
        auto target_indices = selected_indices.device() == target.device() ? selected_indices : selected_indices.to(target.device());
        auto selected_targets = target.index_select(0, target_indices).clone();

        auto flipped = selected_inputs.flip(dims);
        auto augmented_inputs = torch::cat({tensor, flipped}, 0);
        auto augmented_targets = torch::cat({target, selected_targets}, 0);
        return {std::move(augmented_inputs), std::move(augmented_targets)};
    }

    inline std::pair<torch::Tensor, torch::Tensor> Cutout(const torch::Tensor& inputs, const torch::Tensor& targets,
                                                      const std::vector<int64_t>& offsets, const std::vector<int64_t>& sizes,
                                                      double fill_value = 0.0, std::optional<double> frequency = 0.3,
                                                      std::optional<bool> data_augment = true, bool show_progress = true) {
        [[maybe_unused]] const bool show = show_progress;
        if (!Details::augmentation_enabled(data_augment)) {
            return {inputs, targets};
        }
        if (!inputs.defined() || inputs.dim() < 2) {
            return {inputs, targets};
        }
        if (inputs.size(0) != targets.size(0)) {
            throw std::invalid_argument("Inputs and targets must have matching batch dimensions for Cutout augmentation.");
        }

        auto selected_indices = Details::select_indices_by_frequency(inputs.size(0), frequency, inputs.device());
        if (selected_indices.numel() == 0) {
            return {inputs, targets};
        }

        auto selected_inputs = inputs.index_select(0, selected_indices).clone();
        auto target_indices = selected_indices.device() == targets.device() ? selected_indices : selected_indices.to(targets.device());
        auto selected_targets = targets.index_select(0, target_indices).clone();

        auto result = selected_inputs.clone();

        const auto height_dim = result.dim() - 2;
        const auto width_dim = result.dim() - 1;
        const auto height = result.size(height_dim);
        const auto width = result.size(width_dim);

        auto y0 = offsets.size() > 0 ? offsets[0] : int64_t{0};
        auto x0 = offsets.size() > 1 ? offsets[1] : int64_t{0};
        auto h  = sizes.size()   > 0 ? sizes[0]   : height;
        auto w  = sizes.size()   > 1 ? sizes[1]   : width;

        y0 = std::clamp<int64_t>(y0, 0, height);
        x0 = std::clamp<int64_t>(x0, 0, width);
        h  = std::clamp<int64_t>(h,  0, height);
        w  = std::clamp<int64_t>(w,  0, width);

        static thread_local std::mt19937 rng{std::random_device{}()};

        if (offsets.size() > 0 && offsets[0] < 0 && h <= height) {
            std::uniform_int_distribution<int64_t> distribution(0, height - h);
            y0 = distribution(rng);
        }
        if (offsets.size() > 1 && offsets[1] < 0 && w <= width) {
            std::uniform_int_distribution<int64_t> distribution(0, width - w);
            x0 = distribution(rng);
        }

        const auto y1 = std::min<int64_t>(height, y0 + h);
        const auto x1 = std::min<int64_t>(width,  x0 + w);

        if (y0 < y1 && x0 < x1) {
            auto patch = result.slice(height_dim, y0, y1).slice(width_dim, x0, x1);
            if (fill_value == -1.0) {
                auto noise = torch::rand_like(patch);
                patch.copy_(noise);
            } else {
                patch.fill_(fill_value);
            }
        }

        auto augmented_inputs = torch::cat({inputs, result}, 0);
        auto augmented_targets = torch::cat({targets, selected_targets}, 0);
        return {std::move(augmented_inputs), std::move(augmented_targets)};
    }

    inline std::pair<torch::Tensor, torch::Tensor> Shuffle(const torch::Tensor& inputs,
                                                           const torch::Tensor& targets,
                                                           const std::optional<std::uint64_t>& seed = std::nullopt) {
        if (inputs.dim() == 0 || targets.dim() == 0 || inputs.size(0) == 0) {
            return {inputs.clone(), targets.clone()};
        }

        if (inputs.size(0) != targets.size(0)) {
            throw std::invalid_argument("Inputs and targets must contain the same number of samples to shuffle.");
        }

        if (seed.has_value()) {
            torch::manual_seed(static_cast<std::uint64_t>(*seed));
        }

        auto permutation = torch::randperm(inputs.size(0), torch::TensorOptions().dtype(torch::kLong).device(inputs.device()));
        if (permutation.device() != inputs.device()) {
            permutation = permutation.to(inputs.device());
        }

        auto shuffled_inputs = inputs.index_select(0, permutation);
        auto shuffled_targets = targets.index_select(0, permutation);
        return {shuffled_inputs, shuffled_targets};
    }

    inline std::pair<torch::Tensor, torch::Tensor> Fraction(const torch::Tensor& inputs,
                                                            const torch::Tensor& targets,
                                                            float fraction)
    {
        if (!inputs.defined() || !targets.defined()) {
            throw std::invalid_argument("Fraction expects both input and target tensors to be defined.");
        }

        if (inputs.dim() == 0 || targets.dim() == 0) {
            throw std::invalid_argument("Fraction expects tensors with at least one dimension.");
        }

        if (inputs.size(0) != targets.size(0)) {
            throw std::invalid_argument("Inputs and targets must contain the same number of samples to extract a fraction.");
        }

        const auto total_samples = inputs.size(0);
        if (total_samples == 0) {
            return {inputs.clone(), targets.clone()};
        }

        const auto clamped_fraction = std::clamp(fraction, 0.0f, 1.0f);
        auto desired_samples = static_cast<int64_t>(std::round(static_cast<double>(total_samples) * static_cast<double>(clamped_fraction)));
        desired_samples = std::clamp<int64_t>(desired_samples, 0, total_samples);

        if (desired_samples == total_samples) {
            return {inputs.clone(), targets.clone()};
        }

        auto fraction_inputs = inputs.narrow(0, 0, desired_samples).clone();
        auto fraction_targets = targets.narrow(0, 0, desired_samples).clone();
        return {fraction_inputs, fraction_targets};
    }


    inline torch::Tensor Upsample(const torch::Tensor& input,
                                  const std::vector<double>& scale_factors,
                                  torch::nn::functional::InterpolateFuncOptions::mode_t mode = torch::kNearest,
                                  bool align_corners = false,
                                  bool recompute_scale_factor = false) {
        using namespace torch::nn::functional;

        InterpolateFuncOptions options;
        options = options.mode(mode);
        options = options.align_corners(align_corners);
        options = options.recompute_scale_factor(recompute_scale_factor);
        options = options.scale_factor(scale_factors);

        return interpolate(input, options);
    }

    inline torch::Tensor Grayscale(const torch::Tensor& input,
                                   const std::array<double, 3>& weights = {0.2989, 0.5870, 0.1140}) {
        if (input.dim() < 3) {
            throw std::invalid_argument("Grayscale expects a tensor with at least 3 dimensions (C, H, W).");
        }

        auto tensor = input;
        bool had_batch = false;
        if (tensor.dim() == 3) {
            tensor = tensor.unsqueeze(0); // [1, C, H, W]
            had_batch = true;
        }

        if (tensor.size(1) != 3) {
            throw std::invalid_argument("Grayscale expects the channel dimension to have size 3.");
        }

        auto options = tensor.options().dtype(torch::kFloat32);
        auto weight_tensor = torch::from_blob(const_cast<double*>(weights.data()), {3}, torch::TensorOptions().dtype(torch::kDouble));
        weight_tensor = weight_tensor.to(options.dtype()).to(tensor.device()).view({3, 1, 1});

        auto grayscale = (tensor.to(options.dtype()) * weight_tensor).sum(1, true);
        if (had_batch) {
            grayscale = grayscale.squeeze(0);
        }
        return grayscale;
    }
#include "manipulation/augment/augment.hpp"
}

namespace Thot::Data::Check {
    struct ImbalanceReport {
        torch::Tensor train_counts;
        torch::Tensor test_counts;
        torch::Tensor train_distribution;
        torch::Tensor test_distribution;
        double kl_divergence{0.0};
        double train_entropy{0.0};
        double test_entropy{0.0};

        [[nodiscard]] std::string str() const {
            std::ostringstream oss;
            oss << "Class imbalance diagnostics:\n";
            oss << "  Train counts: " << train_counts.to(torch::kCPU) << "\n";
            oss << "  Test counts: " << test_counts.to(torch::kCPU) << "\n";
            oss << "  Train distribution: " << train_distribution.to(torch::kCPU) << "\n";
            oss << "  Test distribution: " << test_distribution.to(torch::kCPU) << "\n";
            oss << "  KL(train || test): " << kl_divergence << "\n";
            oss << "  Entropy(train): " << train_entropy << ", Entropy(test): " << test_entropy;
            return oss.str();
        }
    };

    inline torch::Tensor compute_counts(const torch::Tensor& targets, int64_t num_classes) {
        auto cpu_targets = targets.to(torch::kCPU);
        if (cpu_targets.dim() != 1) {
            cpu_targets = cpu_targets.reshape({-1});
        }
        if (cpu_targets.dtype() != torch::kLong) {
            cpu_targets = cpu_targets.to(torch::kLong);
        }
        return torch::bincount(cpu_targets, {}, num_classes);
    }

    inline torch::Tensor safe_distribution(const torch::Tensor& counts) {
        auto total = counts.sum().item<double>();
        if (total <= 0.0) {
            return torch::zeros_like(counts, torch::TensorOptions().dtype(torch::kDouble));
        }
        return counts.to(torch::kDouble) / total;
    }

    inline double entropy(const torch::Tensor& distribution) {
        auto eps = 1e-12;
        auto clipped = torch::clamp(distribution, eps, 1.0);
        auto entropy_tensor = -(clipped * torch::log(clipped));
        return entropy_tensor.sum().item<double>();
    }

    inline ImbalanceReport Imbalance(const torch::Tensor& train_targets,
                                     const torch::Tensor& test_targets,
                                     std::optional<int64_t> explicit_classes = std::nullopt) {
        const auto train_max = train_targets.numel() > 0
            ? train_targets.to(torch::kCPU).max().item<int64_t>() + 1
            : int64_t{0};
        const auto test_max = test_targets.numel() > 0
            ? test_targets.to(torch::kCPU).max().item<int64_t>() + 1
            : int64_t{0};

        const auto inferred_classes = std::max({
            train_max,
            test_max,
            explicit_classes.value_or(int64_t{0})
        });

        const auto num_classes = std::max<int64_t>(inferred_classes, int64_t{1});

        auto train_counts = compute_counts(train_targets, num_classes);
        auto test_counts = compute_counts(test_targets, num_classes);

        auto train_distribution = safe_distribution(train_counts);
        auto test_distribution = safe_distribution(test_counts);

        auto eps = 1e-12;
        auto clipped_train = torch::clamp(train_distribution, eps, 1.0);
        auto clipped_test = torch::clamp(test_distribution, eps, 1.0);

        auto kl_tensor = clipped_train * (torch::log(clipped_train) - torch::log(clipped_test));
        auto kl_divergence = kl_tensor.sum().item<double>();

        ImbalanceReport report{};
        report.train_counts = train_counts;
        report.test_counts = test_counts;
        report.train_distribution = train_distribution;
        report.test_distribution = test_distribution;
        report.kl_divergence = kl_divergence;
        report.train_entropy = entropy(train_distribution);
        report.test_entropy = entropy(test_distribution);
        return report;
    }

    inline std::string Shuffled(const torch::Tensor& targets, double threshold = 0.95) {
        if (targets.numel() <= 1) {
            return "Dataset order: insufficient data to determine shuffle";
        }

        auto flattened = targets.reshape({-1}).to(torch::kCPU);
        if (flattened.dtype() != torch::kLong && flattened.dtype() != torch::kInt64) {
            flattened = flattened.to(torch::kLong);
        }

        auto runs = std::get<0>(torch::unique_consecutive(flattened));
        const auto run_ratio = static_cast<double>(runs.size(0)) / static_cast<double>(flattened.size(0));

        std::ostringstream oss;
        const bool shuffled = run_ratio >= threshold;
        oss << "Dataset order: " << (shuffled ? "likely shuffled" : "likely grouped")
            << " (unique-consecutive ratio=" << run_ratio << ")";
        return oss.str();
    }

    inline std::vector<int> Size(const torch::Tensor&x, const std::string& title="Tensor Shape") {
        at::IntArrayRef sizes = x.sizes();
        std::cout << title << ": (";
        for (size_t i = 0; i < sizes.size(); i++) {
            std::cout << sizes[i];
            if (i + 1 < sizes.size()) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
        return {x.sizes().begin(), x.sizes().end()};
    }
}

#endif //THOT_MANIPULATION_HPP