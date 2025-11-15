#ifndef THOT_MANIPULATION_AUGMENTATION_HPP
#define THOT_MANIPULATION_AUGMENTATION_HPP
#include <optional>
#include <torch/serialize/input-archive.h>

// TODO: Must rework the file architecture to keep it readable and scalable

namespace Details {
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

    inline torch::Tensor apply_clahe_tile(const torch::Tensor& tile, int64_t bins, double clip_limit) {
        auto cpu_tile = tile.to(torch::kCPU).contiguous();
        auto result = torch::empty_like(cpu_tile);
        auto accessor_in = cpu_tile.accessor<float, 2>();
        auto accessor_out = result.accessor<float, 2>();
        const auto tile_h = cpu_tile.size(0);
        const auto tile_w = cpu_tile.size(1);
        const double area = static_cast<double>(tile_h * tile_w);
        const double clip_value = clip_limit <= 0.0 ? std::numeric_limits<double>::infinity()
                                                    : std::max(1.0, clip_limit * area);
        std::vector<double> histogram(bins, 0.0);
        for (int64_t y = 0; y < tile_h; ++y) {
            for (int64_t x = 0; x < tile_w; ++x) {
                const int64_t bin = std::clamp<int64_t>(
                    static_cast<int64_t>(accessor_in[y][x] * (bins - 1)), 0, bins - 1);
                histogram[bin] += 1.0;
            }
        }
        double excess = 0.0;
        if (std::isfinite(clip_value)) {
            for (auto& value : histogram) {
                if (value > clip_value) {
                    excess += value - clip_value;
                    value = clip_value;
                }
            }
            const double redist = excess / bins;
            for (auto& value : histogram) {
                value += redist;
            }
        }
        std::vector<double> cdf(bins, 0.0);
        double cumulative = 0.0;
        for (int64_t i = 0; i < bins; ++i) {
            cumulative += histogram[i];
            cdf[i] = cumulative;
        }
        if (cdf.back() > 0.0) {
            for (auto& value : cdf) {
                value /= cdf.back();
            }
        }
        for (int64_t y = 0; y < tile_h; ++y) {
            for (int64_t x = 0; x < tile_w; ++x) {
                const int64_t bin = std::clamp<int64_t>(
                    static_cast<int64_t>(accessor_in[y][x] * (bins - 1)), 0, bins - 1);
                accessor_out[y][x] = static_cast<float>(cdf[bin]);
            }
        }
        return result.to(tile.device());
    }
}

inline std::pair<torch::Tensor, torch::Tensor> RandomBrightnessContrast(
    const torch::Tensor& inputs,
    const torch::Tensor& targets,
    double brightness_delta = 0.15,
    double contrast_delta = 0.35,
    std::optional<double> frequency = 0.3,
    std::optional<bool> data_augment = true,
    bool show_progress = true) {
    [[maybe_unused]] const bool show = show_progress;
    auto selection = Details::select_augmented_subset(inputs, targets, frequency, data_augment);
    if (!selection.has_value()) {
        return {inputs, targets};
    }

    auto float_inputs = Details::ensure_float_tensor(selection->inputs);
    auto options = float_inputs.options();
    auto brightness = torch::empty({float_inputs.size(0), 1, 1, 1}, options)
        .uniform_(-brightness_delta, brightness_delta);
    auto contrast = torch::empty({float_inputs.size(0), 1, 1, 1}, options)
        .uniform_(1.0 - contrast_delta, 1.0 + contrast_delta);
    auto adjusted = float_inputs * contrast + brightness;
    const auto [min_value, max_value] = Details::infer_tensor_bounds(selection->inputs);
    adjusted = Details::clamp_to_range(adjusted, min_value, max_value);
    adjusted = adjusted.to(selection->inputs.scalar_type());
    return Details::finalize_augmentation(inputs, targets, std::move(adjusted), std::move(selection->targets));
}

inline std::pair<torch::Tensor, torch::Tensor> CLAHE(
    const torch::Tensor& inputs,
    const torch::Tensor& targets,
    int64_t histogram_bins = 256,
    double clip_limit = 0.01,
    std::pair<int64_t, int64_t> tile_grid = {8, 8},
    std::optional<double> frequency = 0.3,
    std::optional<bool> data_augment = true,
    bool show_progress = true) {
    [[maybe_unused]] const bool show = show_progress;
    auto selection = Details::select_augmented_subset(inputs, targets, frequency, data_augment);
    if (!selection.has_value() || selection->inputs.dim() < 3) {
        return {inputs, targets};
    }

    auto float_inputs = Details::ensure_float_tensor(selection->inputs);
    const auto [min_value, max_value] = Details::infer_tensor_bounds(selection->inputs);
    auto normalized = Details::normalize_unit_range(float_inputs, min_value, max_value);
    const auto batch = normalized.size(0);
    const auto channels = normalized.dim() > 3 ? normalized.size(1) : 1;
    const auto height = normalized.size(normalized.dim() - 2);
    const auto width = normalized.size(normalized.dim() - 1);

    auto result = normalized.clone();
    const int64_t rows = std::max<int64_t>(1, tile_grid.first);
    const int64_t cols = std::max<int64_t>(1, tile_grid.second);
    const int64_t tile_h = std::max<int64_t>(1, height / rows);
    const int64_t tile_w = std::max<int64_t>(1, width / cols);

    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t c = 0; c < channels; ++c) {
            auto channel = normalized.dim() > 3 ? result[n][c] : result[n];
            for (int64_t row = 0; row < rows; ++row) {
                const int64_t y0 = row * tile_h;
                const int64_t y1 = row + 1 == rows ? height : std::min<int64_t>(height, y0 + tile_h);
                for (int64_t col = 0; col < cols; ++col) {
                    const int64_t x0 = col * tile_w;
                    const int64_t x1 = col + 1 == cols ? width : std::min<int64_t>(width, x0 + tile_w);
                    auto tile = channel.slice(-2, y0, y1).slice(-1, x0, x1);
                    auto equalized = Details::apply_clahe_tile(tile, histogram_bins, clip_limit);
                    channel.slice(-2, y0, y1).slice(-1, x0, x1).copy_(equalized);
                }
            }
        }
    }

    auto denormalized = Details::denormalize_unit_range(result, min_value, max_value);
    denormalized = Details::clamp_to_range(denormalized, min_value, max_value);
    denormalized = denormalized.to(selection->inputs.scalar_type());
    return Details::finalize_augmentation(inputs, targets, std::move(denormalized), std::move(selection->targets));
}

inline std::pair<torch::Tensor, torch::Tensor> GridDistortion(
    const torch::Tensor& inputs,
    const torch::Tensor& targets,
    double distort_limit = 0.08,
    int64_t control_points = 5,
    std::optional<double> frequency = 0.3,
    std::optional<bool> data_augment = true,
    bool show_progress = true) {
    [[maybe_unused]] const bool show = show_progress;
    auto selection = Details::select_augmented_subset(inputs, targets, frequency, data_augment);
    if (!selection.has_value() || selection->inputs.dim() < 4) {
        return {inputs, targets};
    }

    auto float_inputs = Details::ensure_float_tensor(selection->inputs);
    const auto device = float_inputs.device();
    const auto batch = float_inputs.size(0);
    const auto height = float_inputs.size(2);
    const auto width = float_inputs.size(3);

    auto base_grid = Details::identity_grid(height, width, device);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto offsets = torch::empty({1, 2, control_points, control_points}, options)
        .uniform_(-distort_limit, distort_limit);
    auto upsampled = torch::nn::functional::interpolate(
        offsets,
        torch::nn::functional::InterpolateFuncOptions()
            .mode(torch::kBilinear)
            .align_corners(true)
            .size(std::vector<int64_t>{height, width}));
    auto grid = base_grid + upsampled.permute({0, 2, 3, 1}).squeeze(0);
    grid = grid.unsqueeze(0).repeat({batch, 1, 1, 1});
    auto warped = Details::grid_sample_image(float_inputs, grid);
    const auto [min_value, max_value] = Details::infer_tensor_bounds(selection->inputs);
    warped = Details::clamp_to_range(warped, min_value, max_value);
    warped = warped.to(selection->inputs.scalar_type());
    return Details::finalize_augmentation(inputs, targets, std::move(warped), std::move(selection->targets));
}

inline std::pair<torch::Tensor, torch::Tensor> OpticalDistortion(
    const torch::Tensor& inputs,
    const torch::Tensor& targets,
    std::pair<double, double> k1_range = {-0.1, 0.1},
    std::pair<double, double> k2_range = {-0.05, 0.05},
    std::optional<double> frequency = 0.3,
    std::optional<bool> data_augment = true,
    bool show_progress = true) {
    [[maybe_unused]] const bool show = show_progress;
    auto selection = Details::select_augmented_subset(inputs, targets, frequency, data_augment);
    if (!selection.has_value() || selection->inputs.dim() < 4) {
        return {inputs, targets};
    }

    auto float_inputs = Details::ensure_float_tensor(selection->inputs);
    const auto device = float_inputs.device();
    const auto batch = float_inputs.size(0);
    const auto height = float_inputs.size(2);
    const auto width = float_inputs.size(3);

    auto base_grid = Details::identity_grid(height, width, device)
        .unsqueeze(0)
        .repeat({batch, 1, 1, 1});
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto k1 = torch::empty({batch, 1, 1}, options).uniform_(k1_range.first, k1_range.second);
    auto k2 = torch::empty({batch, 1, 1}, options).uniform_(k2_range.first, k2_range.second);

    auto x = base_grid.select(-1, 0);
    auto y = base_grid.select(-1, 1);
    auto r2 = x * x + y * y;
    auto factor = 1 + k1 * r2 + k2 * r2 * r2;
    auto distorted_grid = torch::stack({x * factor, y * factor}, -1);
    auto warped = Details::grid_sample_image(float_inputs, distorted_grid);
    const auto [min_value, max_value] = Details::infer_tensor_bounds(selection->inputs);
    warped = Details::clamp_to_range(warped, min_value, max_value);
    warped = warped.to(selection->inputs.scalar_type());
    return Details::finalize_augmentation(inputs, targets, std::move(warped), std::move(selection->targets));
}

inline std::pair<torch::Tensor, torch::Tensor> AtmosphericDrift(
    const torch::Tensor& inputs,
    const torch::Tensor& targets,
    std::array<double, 3> atmospheric_color = {0.9, 0.95, 1.0},
    std::pair<double, double> strength = {0.05, 0.2},
    std::pair<double, double> drift = {0.0, 0.25},
    std::optional<double> frequency = 0.3,
    std::optional<bool> data_augment = true,
    bool show_progress = true) {
    [[maybe_unused]] const bool show = show_progress;
    auto selection = Details::select_augmented_subset(inputs, targets, frequency, data_augment);
    if (!selection.has_value() || selection->inputs.dim() < 4) {
        return {inputs, targets};
    }

    auto float_inputs = Details::ensure_float_tensor(selection->inputs);
    const auto batch = float_inputs.size(0);
    const auto channels = float_inputs.size(1);
    const auto height = float_inputs.size(2);
    auto options = float_inputs.options();

    auto haze_strength = torch::empty({batch, 1, 1, 1}, options)
        .uniform_(strength.first, strength.second);
    auto drift_strength = torch::empty({batch, 1, 1, 1}, options)
        .uniform_(drift.first, drift.second);

    auto gradient = torch::linspace(0.0, 1.0, height, options).view({1, 1, height, 1});
    auto drift_map = gradient * drift_strength;
    auto haze_map = torch::clamp(haze_strength + drift_map, 0.0, 1.0);

    std::vector<double> color_vec(channels, atmospheric_color[0]);
    for (int64_t c = 0; c < channels && c < static_cast<int64_t>(atmospheric_color.size()); ++c) {
        color_vec[c] = atmospheric_color[c];
    }
    auto color = torch::tensor(color_vec, options).view({1, channels, 1, 1});

    auto adjusted = float_inputs * (1.0 - haze_map) + color * haze_map;
    const auto [min_value, max_value] = Details::infer_tensor_bounds(selection->inputs);
    adjusted = Details::clamp_to_range(adjusted, min_value, max_value);
    adjusted = adjusted.to(selection->inputs.scalar_type());
    return Details::finalize_augmentation(inputs, targets, std::move(adjusted), std::move(selection->targets));
}

inline std::pair<torch::Tensor, torch::Tensor> SunAngleJitter(
    const torch::Tensor& inputs,
    const torch::Tensor& targets,
    double base_angle_degrees = 45.0,
    double jitter_degrees = 20.0,
    std::pair<double, double> shadow_strength = {0.15, 0.45},
    std::optional<double> frequency = 0.3,
    std::optional<bool> data_augment = true,
    bool show_progress = true) {
    [[maybe_unused]] const bool show = show_progress;
    auto selection = Details::select_augmented_subset(inputs, targets, frequency, data_augment);
    if (!selection.has_value() || selection->inputs.dim() < 4) {
        return {inputs, targets};
    }

    auto float_inputs = Details::ensure_float_tensor(selection->inputs);
    const auto batch = float_inputs.size(0);
    const auto height = float_inputs.size(2);
    const auto width = float_inputs.size(3);
    auto options = float_inputs.options();

    auto base_grid = Details::identity_grid(height, width, float_inputs.device()).unsqueeze(0);
    auto jitter = torch::empty({batch, 1}, options).uniform_(-jitter_degrees, jitter_degrees);
    constexpr double kPi = 3.14159265358979323846;
    auto total_angle = (base_angle_degrees + jitter) * (kPi / 180.0);
    auto direction = torch::stack({torch::cos(total_angle), torch::sin(total_angle)}, -1);
    auto mask = (base_grid * direction.view({batch, 1, 1, 2})).sum(-1, true);
    auto min_vals = mask.amin({1, 2, 3}, true);
    auto max_vals = mask.amax({1, 2, 3}, true);
    mask = (mask - min_vals) / (max_vals - min_vals + 1e-6);
    auto strengths = torch::empty({batch, 1, 1, 1}, options)
        .uniform_(shadow_strength.first, shadow_strength.second);
    auto attenuation = 1.0 - mask * strengths;
    auto adjusted = float_inputs * attenuation;
    const auto [min_value, max_value] = Details::infer_tensor_bounds(selection->inputs);
    adjusted = Details::clamp_to_range(adjusted, min_value, max_value);
    adjusted = adjusted.to(selection->inputs.scalar_type());
    return Details::finalize_augmentation(inputs, targets, std::move(adjusted), std::move(selection->targets));
}

inline std::pair<torch::Tensor, torch::Tensor> CloudOcclusion(
    const torch::Tensor& inputs,
    const torch::Tensor& targets,
    int64_t max_clouds = 4,
    double coverage = 0.35,
    double softness = 6.0,
    std::array<double, 3> cloud_color = {1.0, 1.0, 1.0},
    std::optional<double> frequency = 0.3,
    std::optional<bool> data_augment = true,
    bool show_progress = true) {
    [[maybe_unused]] const bool show = show_progress;
    auto selection = Details::select_augmented_subset(inputs, targets, frequency, data_augment);
    if (!selection.has_value() || selection->inputs.dim() < 4) {
        return {inputs, targets};
    }

    auto float_inputs = Details::ensure_float_tensor(selection->inputs);
    const auto batch = float_inputs.size(0);
    const auto channels = float_inputs.size(1);
    const auto height = float_inputs.size(2);
    const auto width = float_inputs.size(3);
    auto options = float_inputs.options();

    auto y_coords = torch::linspace(0.0, 1.0, height, options).view({height, 1}).repeat({1, width});
    auto x_coords = torch::linspace(0.0, 1.0, width, options).view({1, width}).repeat({height, 1});

    auto mask = torch::ones({batch, 1, height, width}, options);
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<double> dist01(0.0, 1.0);

    for (int64_t n = 0; n < batch; ++n) {
        auto sample_mask = mask[n][0];
        const int clouds = std::max<int>(1, static_cast<int>(std::round(dist01(rng) * max_clouds)));
        for (int i = 0; i < clouds; ++i) {
            const double cx = dist01(rng);
            const double cy = dist01(rng);
            const double rx = 0.1 + dist01(rng) * 0.4;
            const double ry = 0.1 + dist01(rng) * 0.4;
            const double alpha = coverage * dist01(rng);
            auto distance = torch::pow((x_coords - cx) / rx, 2) + torch::pow((y_coords - cy) / ry, 2);
            auto contribution = torch::exp(-distance * softness);
            sample_mask = torch::minimum(sample_mask, 1.0 - alpha * contribution);
        }
        mask[n][0].copy_(sample_mask);
    }

    std::vector<double> color_vec(channels, cloud_color[0]);
    for (int64_t c = 0; c < channels && c < static_cast<int64_t>(cloud_color.size()); ++c) {
        color_vec[c] = cloud_color[c];
    }
    auto color = torch::tensor(color_vec, options).view({1, channels, 1, 1});
    auto occluded = float_inputs * mask + color * (1.0 - mask);
    const auto [min_value, max_value] = Details::infer_tensor_bounds(selection->inputs);
    occluded = Details::clamp_to_range(occluded, min_value, max_value);
    occluded = occluded.to(selection->inputs.scalar_type());
    return Details::finalize_augmentation(inputs, targets, std::move(occluded), std::move(selection->targets));
}

inline std::pair<torch::Tensor, torch::Tensor> ChromaticAberration(
    const torch::Tensor& inputs,
    const torch::Tensor& targets,
    double max_shift_pixels = 2.0,
    std::optional<double> frequency = 0.3,
    std::optional<bool> data_augment = true,
    bool show_progress = true) {
    [[maybe_unused]] const bool show = show_progress;
    auto selection = Details::select_augmented_subset(inputs, targets, frequency, data_augment);
    if (!selection.has_value() || selection->inputs.dim() < 4 || selection->inputs.size(1) < 3) {
        return {inputs, targets};
    }

    auto float_inputs = Details::ensure_float_tensor(selection->inputs);
    const auto batch = float_inputs.size(0);
    const auto channels = float_inputs.size(1);
    const auto height = float_inputs.size(2);
    const auto width = float_inputs.size(3);
    auto options = float_inputs.options();

    auto base_grid = Details::identity_grid(height, width, float_inputs.device());
    const double norm_shift_x = max_shift_pixels * 2.0 / std::max<int64_t>(1, width - 1);
    const double norm_shift_y = max_shift_pixels * 2.0 / std::max<int64_t>(1, height - 1);

    std::vector<torch::Tensor> warped_channels;
    warped_channels.reserve(channels);

    for (int64_t c = 0; c < channels; ++c) {
        auto channel = float_inputs.select(1, c).unsqueeze(1);
        auto shifts = torch::empty({batch, 2}, options)
            .uniform_(-1.0, 1.0);
        shifts.select(1, 0).mul_(norm_shift_x);
        shifts.select(1, 1).mul_(norm_shift_y);
        auto grid = base_grid.unsqueeze(0).repeat({batch, 1, 1, 1}).clone();
        grid.select(-1, 0) += shifts.select(1, 0).view({batch, 1, 1});
        grid.select(-1, 1) += shifts.select(1, 1).view({batch, 1, 1});
        auto warped = Details::grid_sample_image(channel, grid);
        warped_channels.push_back(warped);
    }

    auto aberrated = torch::cat(warped_channels, 1);
    const auto [min_value, max_value] = Details::infer_tensor_bounds(selection->inputs);
    aberrated = Details::clamp_to_range(aberrated, min_value, max_value);
    aberrated = aberrated.to(selection->inputs.scalar_type());
    return Details::finalize_augmentation(inputs, targets, std::move(aberrated), std::move(selection->targets));
}

#endif // THOT_MANIPULATION_AUGMENTATION_HPP