#ifndef OMNI_DATA_TRANSFORM_AUGMENTATION_ATMOSPHERIC_DRIFT_HPP
#define OMNI_DATA_TRANSFORM_AUGMENTATION_ATMOSPHERIC_DRIFT_HPP

#include <array>
#include <optional>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "common.hpp"

namespace Omni::Data::Transform::Augmentation {
    namespace Options {
        struct AtmosphericDriftOptions {
            std::array<double, 3> atmospheric_color = {0.9, 0.95, 1.0};
            std::pair<double, double> strength = {0.05, 0.2};
            std::pair<double, double> drift = {0.0, 0.25};
            std::optional<double> frequency = 0.3;
            std::optional<bool> data_augment = true;
            bool show_progress = true;
        };
    }
    inline std::pair<torch::Tensor, torch::Tensor> AtmosphericDrift(const torch::Tensor& inputs, const torch::Tensor& targets, Options::AtmosphericDriftOptions opt) {
        [[maybe_unused]] const bool show = opt.show_progress;
        auto selection = Details::select_augmented_subset(inputs, targets, opt.frequency, opt.data_augment);
        if (!selection.has_value() || selection->inputs.dim() < 4) {
            return {inputs, targets};
        }

        auto float_inputs = Details::ensure_float_tensor(selection->inputs);
        const auto batch = float_inputs.size(0);
        const auto channels = float_inputs.size(1);
        const auto height = float_inputs.size(2);
        auto options = float_inputs.options();

        auto haze_strength = torch::empty({batch, 1, 1, 1}, options)
            .uniform_(opt.strength.first, opt.strength.second);
        auto drift_strength = torch::empty({batch, 1, 1, 1}, options)
            .uniform_(opt.drift.first, opt.drift.second);

        auto gradient = torch::linspace(0.0, 1.0, height, options).view({1, 1, height, 1});
        auto drift_map = gradient * drift_strength;
        auto haze_map = torch::clamp(haze_strength + drift_map, 0.0, 1.0);

        std::vector<double> color_vec(channels, opt.atmospheric_color[0]);
        for (int64_t c = 0; c < channels && c < static_cast<int64_t>(opt.atmospheric_color.size()); ++c) {
            color_vec[c] = opt.atmospheric_color[c];
        }
        auto color = torch::tensor(color_vec, options).view({1, channels, 1, 1});

        auto adjusted = float_inputs * (1.0 - haze_map) + color * haze_map;
        const auto [min_value, max_value] = Details::infer_tensor_bounds(selection->inputs);
        adjusted = Details::clamp_to_range(adjusted, min_value, max_value);
        adjusted = adjusted.to(selection->inputs.scalar_type());
        return Details::finalize_augmentation(inputs, targets, std::move(adjusted), std::move(selection->targets));
    }
}

#endif // OMNI_DATA_TRANSFORM_AUGMENTATION_ATMOSPHERIC_DRIFT_HPP