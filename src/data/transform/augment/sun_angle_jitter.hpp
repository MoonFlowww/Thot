#ifndef THOT_DATA_TRANSFORMS_AUGMENTATION_SUN_ANGLE_JITTER_HPP
#define THOT_DATA_TRANSFORMS_AUGMENTATION_SUN_ANGLE_JITTER_HPP

#include <cmath>
#include <optional>
#include <utility>

#include <torch/torch.h>

#include "common.hpp"

namespace Thot::Data::Transforms::Augmentation {
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
}

#endif // THOT_DATA_TRANSFORMS_AUGMENTATION_SUN_ANGLE_JITTER_HPP