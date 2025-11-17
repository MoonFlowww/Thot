#ifndef THOT_DATA_TRANSFORM_AUGMENTATION_COMMON_HPP
#define THOT_DATA_TRANSFORM_AUGMENTATION_COMMON_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>

#include <torch/nn/functional.h>
#include <torch/torch.h>

namespace Thot::Data::Transform::Augmentation::Details {
    inline bool augmentation_enabled(const std::optional<bool>& data_augment) {
        return !(data_augment.has_value() && !data_augment.value());
    }

    inline double clamp_frequency(const std::optional<double>& frequency) {
        if (!frequency.has_value()) {
            return 1.0;
        }
        return std::clamp(frequency.value(), 0.0, 1.0);
    }

    inline torch::Tensor select_indices_by_frequency(int64_t batch_size,
                                                     const std::optional<double>& frequency,
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
        return torch::arange(0, batch_size, stride, long_options);
    }

    struct AugmentationSelection {
        torch::Tensor inputs;
        torch::Tensor targets;
    };

    inline std::optional<AugmentationSelection> select_augmented_subset(
        const torch::Tensor& inputs,
        const torch::Tensor& targets,
        const std::optional<double>& frequency,
        const std::optional<bool>& data_augment) {
        if (!augmentation_enabled(data_augment)) {
            return std::nullopt;
        }
        if (!inputs.defined() || !targets.defined() || inputs.dim() == 0 || targets.dim() == 0) {
            return std::nullopt;
        }
        if (inputs.size(0) != targets.size(0)) {
            throw std::invalid_argument("Inputs and targets must share the batch dimension for augmentation.");
        }
        auto indices = select_indices_by_frequency(inputs.size(0), frequency, inputs.device());
        if (indices.numel() == 0) {
            return std::nullopt;
        }
        auto selected_inputs = inputs.index_select(0, indices).clone();
        auto target_indices = indices.device() == targets.device() ? indices : indices.to(targets.device());
        auto selected_targets = targets.index_select(0, target_indices).clone();
        return AugmentationSelection{std::move(selected_inputs), std::move(selected_targets)};
    }

    inline std::pair<torch::Tensor, torch::Tensor> finalize_augmentation(
        const torch::Tensor& inputs,
        const torch::Tensor& targets,
        torch::Tensor&& augmented_inputs,
        torch::Tensor&& augmented_targets) {
        auto concatenated_inputs = torch::cat({inputs, augmented_inputs}, 0);
        auto concatenated_targets = torch::cat({targets, augmented_targets}, 0);
        return {std::move(concatenated_inputs), std::move(concatenated_targets)};
    }

    inline std::pair<double, double> infer_tensor_bounds(const torch::Tensor& tensor) {
        auto min_tensor = tensor.amin();
        auto max_tensor = tensor.amax();
        const double min_value = min_tensor.item<double>();
        double max_value = max_tensor.item<double>();
        if (std::abs(max_value - min_value) < 1e-12) {
            max_value = min_value + 1.0;
        }
        return {min_value, max_value};
    }

    inline torch::Tensor clamp_to_range(const torch::Tensor& tensor, double min_value, double max_value) {
        return torch::clamp(tensor, min_value, max_value);
    }

    inline torch::Tensor normalize_unit_range(const torch::Tensor& tensor, double min_value, double max_value) {
        const double span = std::max(1e-12, max_value - min_value);
        return (tensor - min_value) / span;
    }

    inline torch::Tensor denormalize_unit_range(const torch::Tensor& tensor, double min_value, double max_value) {
        return tensor * (max_value - min_value) + min_value;
    }

    inline torch::Tensor ensure_float_tensor(const torch::Tensor& tensor) {
        if (tensor.scalar_type() == torch::kFloat32 || tensor.scalar_type() == torch::kFloat64) {
            return tensor.clone();
        }
        return tensor.to(torch::kFloat32);
    }

    inline torch::Tensor identity_grid(int64_t height, int64_t width, const torch::Device& device) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        auto ys = torch::linspace(-1.0, 1.0, height, options);
        auto xs = torch::linspace(-1.0, 1.0, width, options);
        auto grid_y = ys.view({height, 1}).repeat({1, width});
        auto grid_x = xs.view({1, width}).repeat({height, 1});
        return torch::stack({grid_x, grid_y}, -1);
    }

    inline torch::Tensor grid_sample_image(const torch::Tensor& image, const torch::Tensor& grid) {
        using namespace torch::nn::functional;
        GridSampleFuncOptions options;
        options = options.mode(torch::kBilinear);
        options = options.align_corners(true);
        options = options.padding_mode(torch::kReflection);
        return grid_sample(image, grid, options);
    }
    inline std::optional<torch::Tensor> maybe_warp_targets_with_grid(const torch::Tensor& targets,
                                                                     const torch::Tensor& grid) {
        if (!targets.defined() || grid.dim() != 4) {
            return std::nullopt;
        }

        auto rank = targets.dim();
        if (rank < 3) {
            return std::nullopt;
        }

        bool added_channel_dim = false;
        auto working = targets;
        if (rank == 3) {
            working = working.unsqueeze(1);
            added_channel_dim = true;
        } else if (rank != 4) {
            return std::nullopt;
        }

        const auto target_height = working.size(-2);
        const auto target_width = working.size(-1);
        if (target_height != grid.size(1) || target_width != grid.size(2)) {
            return std::nullopt;
        }

        auto float_targets = ensure_float_tensor(working);
        auto warped = grid_sample_image(float_targets, grid);
        if (added_channel_dim) {
            warped = warped.squeeze(1);
        }
        if (warped.scalar_type() != targets.scalar_type()) {
            warped = warped.to(targets.scalar_type());
        }

        return warped;
    }
}

#endif // THOT_DATA_TRANSFORM_AUGMENTATION_COMMON_HPP