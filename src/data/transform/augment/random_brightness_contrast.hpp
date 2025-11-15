#ifndef THOT_DATA_TRANSFORMS_AUGMENTATION_RANDOM_BRIGHTNESS_CONTRAST_HPP
#define THOT_DATA_TRANSFORMS_AUGMENTATION_RANDOM_BRIGHTNESS_CONTRAST_HPP

#include <optional>

#include <torch/torch.h>

#include "common.hpp"

namespace Thot::Data::Transforms::Augmentation {
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
}

#endif // THOT_DATA_TRANSFORMS_AUGMENTATION_RANDOM_BRIGHTNESS_CONTRAST_HPP