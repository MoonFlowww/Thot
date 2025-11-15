#ifndef THOT_DATA_TRANSFORMS_AUGMENTATION_CLAHE_HPP
#define THOT_DATA_TRANSFORMS_AUGMENTATION_CLAHE_HPP

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "common.hpp"

namespace Thot::Data::Transforms::Augmentation {
    namespace Details {
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
}

#endif // THOT_DATA_TRANSFORMS_AUGMENTATION_CLAHE_HPP