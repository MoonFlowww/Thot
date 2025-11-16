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

        const bool random_y = opt.offsets.size() > 0 && opt.offsets[0] < 0 && height > 0;
        const bool random_x = opt.offsets.size() > 1 && opt.offsets[1] < 0 && width > 0;

        if (random_y) {
            std::uniform_int_distribution<int64_t> distribution(0, height - 1);
            y0 = distribution(rng);
        }
        if (random_x) {
            std::uniform_int_distribution<int64_t> distribution(0, width - 1);
            x0 = distribution(rng);
        }

        const bool requires_toroidal = random_y || random_x;

        if (!requires_toroidal) {
            const auto y1 = std::min<int64_t>(height, y0 + h);
            const auto x1 = std::min<int64_t>(width,  x0 + w);

            if (y0 < y1 && x0 < x1) {
                auto patch = result.slice(height_dim, y0, y1).slice(width_dim, x0, x1);
                if (opt.fill_value == -1.0) {
                    auto noise = torch::randn_like(patch);
                    patch.copy_(noise);
                } else {
                    patch.fill_(opt.fill_value);
                }
            }
        } else {
            auto effective_h = std::min<int64_t>(h, height);
            auto effective_w = std::min<int64_t>(w, width);

            if (!random_y) {
                effective_h = std::min<int64_t>(effective_h, height - y0);
            }
            if (!random_x) {
                effective_w = std::min<int64_t>(effective_w, width - x0);
            }

            if (effective_h > 0 && effective_w > 0) {
                std::vector<int64_t> y_coords(static_cast<std::size_t>(effective_h));
                std::vector<int64_t> x_coords(static_cast<std::size_t>(effective_w));

                for (int64_t i = 0; i < effective_h; ++i) {
                    auto pos = y0 + i;
                    if (random_y) {
                        pos %= height;
                    }
                    y_coords[static_cast<std::size_t>(i)] = pos;
                }
                for (int64_t i = 0; i < effective_w; ++i) {
                    auto pos = x0 + i;
                    if (random_x) {
                        pos %= width;
                    }
                    x_coords[static_cast<std::size_t>(i)] = pos;
                }

                auto options = torch::TensorOptions().dtype(torch::kLong).device(result.device());
                auto options_bool = torch::TensorOptions().dtype(torch::kBool).device(result.device());
                auto mask = torch::zeros({height, width}, options_bool);
                auto y_tensor = torch::tensor(y_coords, options);
                auto x_tensor = torch::tensor(x_coords, options);

                if (y_tensor.numel() > 0 && x_tensor.numel() > 0) {
                    mask.index_put_({y_tensor.unsqueeze(-1), x_tensor.unsqueeze(0)}, true);

                    auto mask_view_shape = std::vector<int64_t>(result.dim(), 1);
                    mask_view_shape[height_dim] = height;
                    mask_view_shape[width_dim] = width;

                    auto mask_expanded = mask.view(mask_view_shape).expand(result.sizes());

                    if (opt.fill_value == -1.0) {
                        auto noise = torch::randn_like(result);
                        result = torch::where(mask_expanded, noise, result);
                    } else {
                        result.masked_fill_(mask_expanded, opt.fill_value);
                    }
                }
            }
        }

        auto augmented_inputs = torch::cat({inputs, result}, 0);
        auto augmented_targets = torch::cat({targets, selected_targets}, 0);
        return {std::move(augmented_inputs), std::move(augmented_targets)};
    }
}

#endif // THOT_DATA_TRANSFORMS_AUGMENTATION_CUTOUT_HPP