#ifndef OMNI_DATA_TRANSFORM_AUGMENTATION_GRID_DISTORTION_HPP
#define OMNI_DATA_TRANSFORM_AUGMENTATION_GRID_DISTORTION_HPP

#include <cstdint>
#include <optional>
#include <vector>

#include <torch/nn/functional.h>
#include <torch/torch.h>

#include "common.hpp"

namespace Omni::Data::Transform::Augmentation {
    namespace Options {
        struct GridDistortionOptions {
            double distort_limit = 0.08;
            int64_t control_points = 5;
            std::optional<double> frequency = 0.3;
            std::optional<bool> data_augment = true;
            bool show_progress = true;
        };
    }
    inline std::pair<torch::Tensor, torch::Tensor> GridDistortion(const torch::Tensor& inputs, const torch::Tensor& targets, Options::GridDistortionOptions opt) {
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

        auto base_grid = Details::identity_grid(height, width, device);
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        auto offsets = torch::empty({1, 2, opt.control_points, opt.control_points}, options)
            .uniform_(-opt.distort_limit, opt.distort_limit);
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

        auto warped_targets = Details::maybe_warp_targets_with_grid(selection->targets, grid);
        auto augmented_targets = warped_targets.has_value() ? std::move(*warped_targets)
                                                           : std::move(selection->targets);
        return Details::finalize_augmentation(inputs, targets, std::move(warped), std::move(augmented_targets));
    }
}

#endif // OMNI_DATA_TRANSFORM_AUGMENTATION_GRID_DISTORTION_HPP