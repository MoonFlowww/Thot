#ifndef THOT_DATA_TRANSFORM_AUGMENTATION_CLOUD_OCCLUSION_HPP
#define THOT_DATA_TRANSFORM_AUGMENTATION_CLOUD_OCCLUSION_HPP

#include <array>
#include <optional>
#include <random>
#include <vector>

#include <torch/torch.h>

#include "common.hpp"

namespace Thot::Data::Transform::Augmentation {
    namespace Options {
        struct CloudOcclusionOptions {
            int64_t max_clouds = 4;
            double coverage = 0.35;
            double softness = 6.0;
            std::array<double, 3> cloud_color = {1.0, 1.0, 1.0};
            std::optional<double> frequency = 0.3;
            std::optional<bool> data_augment = true;
            bool show_progress = true;
        };
    }
    inline std::pair<torch::Tensor, torch::Tensor> CloudOcclusion(const torch::Tensor& inputs, const torch::Tensor& targets, Options::CloudOcclusionOptions opt) {
        [[maybe_unused]] const bool show = opt.show_progress;
        auto selection = Details::select_augmented_subset(inputs, targets, opt.frequency, opt.data_augment);
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
            const int clouds = std::max<int>(1, static_cast<int>(std::round(dist01(rng) * opt.max_clouds)));
            for (int i = 0; i < clouds; ++i) {
                const double cx = dist01(rng);
                const double cy = dist01(rng);
                const double rx = 0.1 + dist01(rng) * 0.4;
                const double ry = 0.1 + dist01(rng) * 0.4;
                const double alpha = opt.coverage * dist01(rng);
                auto distance = torch::pow((x_coords - cx) / rx, 2) + torch::pow((y_coords - cy) / ry, 2);
                auto contribution = torch::exp(-distance * opt.softness);
                sample_mask = torch::minimum(sample_mask, 1.0 - alpha * contribution);
            }
            mask[n][0].copy_(sample_mask);
        }

        std::vector<double> color_vec(channels, opt.cloud_color[0]);
        for (int64_t c = 0; c < channels && c < static_cast<int64_t>(opt.cloud_color.size()); ++c) {
            color_vec[c] = opt.cloud_color[c];
        }
        auto color = torch::tensor(color_vec, options).view({1, channels, 1, 1});
        auto occluded = float_inputs * mask + color * (1.0 - mask);
        const auto [min_value, max_value] = Details::infer_tensor_bounds(selection->inputs);
        occluded = Details::clamp_to_range(occluded, min_value, max_value);
        occluded = occluded.to(selection->inputs.scalar_type());
        return Details::finalize_augmentation(inputs, targets, std::move(occluded), std::move(selection->targets));
    }
}

#endif // THOT_DATA_TRANSFORM_AUGMENTATION_CLOUD_OCCLUSION_HPP