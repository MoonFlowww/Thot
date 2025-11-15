#ifndef THOT_DATA_TRANSFORMS_AUGMENTATION_CUTOUT_HPP
#define THOT_DATA_TRANSFORMS_AUGMENTATION_CUTOUT_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "common.hpp"

namespace Thot::Data::Transforms::Augmentation {
    namespace Options {
        struct CutoutOptions {
            const std::vector<int64_t>& offsets;
            const std::vector<int64_t>& sizes;
            double fill_value = 0.0;
            std::optional<double> frequency = 0.3;
            std::optional<bool> data_augment = true;
            bool show_progress = true;
        };
    }
    inline std::pair<torch::Tensor, torch::Tensor> Cutout(const torch::Tensor& inputs, const torch::Tensor& targets, Options::CutoutOptions opt) {
        [[maybe_unused]] const bool show = opt.show_progress;
        if (!Details::augmentation_enabled(opt.data_augment)) {
            return {inputs, targets};
        }
        if (!inputs.defined() || inputs.dim() < 2) {
            return {inputs, targets};
        }
        if (inputs.size(0) != targets.size(0)) {
            throw std::invalid_argument("Inputs and targets must have matching batch dimensions for Cutout augmentation.");
        }

        auto selected_indices = Details::select_indices_by_frequency(inputs.size(0), opt.frequency, inputs.device());
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

        auto y0 = opt.offsets.size() > 0 ? opt.offsets[0] : int64_t{0};
        auto x0 = opt.offsets.size() > 1 ? opt.offsets[1] : int64_t{0};
        auto h  = opt.sizes.size()   > 0 ? opt.sizes[0]   : height;
        auto w  = opt.sizes.size()   > 1 ? opt.sizes[1]   : width;

        y0 = std::clamp<int64_t>(y0, 0, height);
        x0 = std::clamp<int64_t>(x0, 0, width);
        h  = std::clamp<int64_t>(h,  0, height);
        w  = std::clamp<int64_t>(w,  0, width);

        static thread_local std::mt19937 rng{std::random_device{}()};

        if (opt.offsets.size() > 0 && opt.offsets[0] < 0 && h <= height) {
            std::uniform_int_distribution<int64_t> distribution(0, height - h);
            y0 = distribution(rng);
        }
        if (opt.offsets.size() > 1 && opt.offsets[1] < 0 && w <= width) {
            std::uniform_int_distribution<int64_t> distribution(0, width - w);
            x0 = distribution(rng);
        }

        const auto y1 = std::min<int64_t>(height, y0 + h);
        const auto x1 = std::min<int64_t>(width,  x0 + w);

        if (y0 < y1 && x0 < x1) {
            auto patch = result.slice(height_dim, y0, y1).slice(width_dim, x0, x1);
            if (opt.fill_value == -1.0) {
                auto noise = torch::rand_like(patch);
                patch.copy_(noise);
            } else {
                patch.fill_(opt.fill_value);
            }
        }

        auto augmented_inputs = torch::cat({inputs, result}, 0);
        auto augmented_targets = torch::cat({targets, selected_targets}, 0);
        return {std::move(augmented_inputs), std::move(augmented_targets)};
    }
}

#endif // THOT_DATA_TRANSFORMS_AUGMENTATION_CUTOUT_HPP