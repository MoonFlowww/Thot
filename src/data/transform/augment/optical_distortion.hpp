#ifndef THOT_DATA_TRANSFORMS_AUGMENTATION_OPTICAL_DISTORTION_HPP
#define THOT_DATA_TRANSFORMS_AUGMENTATION_OPTICAL_DISTORTION_HPP

#include <optional>
#include <utility>

#include <torch/torch.h>

#include "common.hpp"

namespace Thot::Data::Transforms::Augmentation {
    namespace Options {
        struct OpticalDistortionOptions {
            std::pair<double, double> k1_range = {-0.1, 0.1};
            std::pair<double, double> k2_range = {-0.05, 0.05};
            std::optional<double> frequency = 0.3;
            std::optional<bool> data_augment = true;
            bool show_progress = true;
        };
    }
    inline std::pair<torch::Tensor, torch::Tensor> OpticalDistortion(
        const torch::Tensor& inputs,
        const torch::Tensor& targets, Options::OpticalDistortionOptions opt) {
        [[maybe_unused]] const bool show = opt.show_progress;
        auto selection = Details::select_augmented_subset(inputs, targets, opt.frequency, opt.data_augment);
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
        auto k1 = torch::empty({batch, 1, 1}, options).uniform_(opt.k1_range.first, opt.k1_range.second);
        auto k2 = torch::empty({batch, 1, 1}, options).uniform_(opt.k2_range.first, opt.k2_range.second);

        auto x = base_grid.select(-1, 0);
        auto y = base_grid.select(-1, 1);
        auto r2 = x * x + y * y;
        auto factor = 1 + k1 * r2 + k2 * r2 * r2;
        auto distorted_grid = torch::stack({x * factor, y * factor}, -1);
        auto warped = Details::grid_sample_image(float_inputs, distorted_grid);
        const auto [min_value, max_value] = Details::infer_tensor_bounds(selection->inputs);
        warped = Details::clamp_to_range(warped, min_value, max_value);
        warped = warped.to(selection->inputs.scalar_type());
        auto warped_targets = Details::maybe_warp_targets_with_grid(selection->targets, distorted_grid);
        auto augmented_targets = warped_targets.has_value() ? std::move(*warped_targets) : std::move(selection->targets);
        return Details::finalize_augmentation(inputs, targets, std::move(warped), std::move(augmented_targets));
    }
}

#endif // THOT_DATA_TRANSFORMS_AUGMENTATION_OPTICAL_DISTORTION_HPP