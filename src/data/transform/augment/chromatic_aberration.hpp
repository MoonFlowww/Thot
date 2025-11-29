#ifndef Nott_DATA_TRANSFORM_AUGMENTATION_CHROMATIC_ABERRATION_HPP
#define Nott_DATA_TRANSFORM_AUGMENTATION_CHROMATIC_ABERRATION_HPP

#include <cstdint>
#include <optional>
#include <vector>

#include <torch/torch.h>

#include "common.hpp"

namespace Nott::Data::Transform::Augmentation {
    namespace Options {
         struct ChromaticAberrationOptions {
             double max_shift_pixels = 2.0;
             std::optional<double> frequency = 0.3;
             std::optional<bool> data_augment = true;
             bool show_progress = true;
         };
    }
    inline std::pair<torch::Tensor, torch::Tensor> ChromaticAberration(
        const torch::Tensor& inputs,
        const torch::Tensor& targets, Options::ChromaticAberrationOptions opt) {
        [[maybe_unused]] const bool show = opt.show_progress;
        auto selection = Details::select_augmented_subset(inputs, targets, opt.frequency, opt.data_augment);
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
        const double norm_shift_x = opt.max_shift_pixels * 2.0 / std::max<int64_t>(1, width - 1);
        const double norm_shift_y = opt.max_shift_pixels * 2.0 / std::max<int64_t>(1, height - 1);

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
}

#endif // Nott_DATA_TRANSFORM_AUGMENTATION_CHROMATIC_ABERRATION_HPP