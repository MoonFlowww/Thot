#ifndef THOT_DATA_TRANSFORMS_AUGMENTATION_FLIP_HPP
#define THOT_DATA_TRANSFORMS_AUGMENTATION_FLIP_HPP

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "common.hpp"

namespace Thot::Data::Transforms::Augmentation {
    namespace Options {
        struct FlipOptions {
            const std::vector<std::string>& axes;
            std::optional<double> frequency = 0.3;
            std::optional<bool> data_augment = true;
            bool show_progress = true;
        };
    }
    namespace Details {
        inline int64_t axis_token_to_dim(const std::string& token, int64_t tensor_dim) {
            if (token.empty()) {
                throw std::invalid_argument("Flip axis tokens must not be empty.");
            }

            try {
                std::size_t processed = 0;
                const auto value = std::stoll(token, &processed, 10);
                if (processed == token.size()) {
                    auto normalized = value;
                    if (normalized < 0) {
                        normalized += tensor_dim;
                    }
                    if (normalized < 0 || normalized >= tensor_dim) {
                        throw std::out_of_range("Flip axis index out of range.");
                    }
                    return normalized;
                }
            } catch (const std::invalid_argument&) {
            } catch (const std::out_of_range&) {
                throw std::out_of_range("Flip axis index out of range.");
            }

            std::string lowered = token;
            std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });

            int64_t offset = 0;
            if (lowered == "x") {
                offset = -1;
            } else if (lowered == "y") {
                offset = -2;
            } else if (lowered == "z") {
                offset = -3;
            } else {
                throw std::invalid_argument("Unsupported flip axis token: " + token);
            }

            const auto dim_index = tensor_dim + offset;
            if (dim_index < 0 || dim_index >= tensor_dim) {
                throw std::out_of_range("Flip axis token is incompatible with tensor rank.");
            }

            return dim_index;
        }

        inline std::vector<int64_t> parse_flip_axes(const std::vector<std::string>& axes, int64_t tensor_dim) {
            std::vector<int64_t> dims;
            dims.reserve(axes.size());
            for (const auto& axis : axes) {
                dims.push_back(axis_token_to_dim(axis, tensor_dim));
            }
            return dims;
        }
    }

    inline std::pair<torch::Tensor, torch::Tensor> Flip(
        const torch::Tensor& tensor,
        const torch::Tensor& target, Options::FlipOptions opt) {
        [[maybe_unused]] const bool show = opt.show_progress;
        if (!Details::augmentation_enabled(opt.data_augment)) {
            return {tensor, target};
        }
        if (!tensor.defined() || !target.defined() || tensor.dim() == 0 || target.dim() == 0) {
            return {tensor, target};
        }
        if (tensor.size(0) != target.size(0)) {
            throw std::invalid_argument("Inputs and targets must have matching batch dimensions for Flip augmentation.");
        }
        if (opt.axes.empty()) {
            return {tensor, target};
        }

        const auto dims = Details::parse_flip_axes(opt.axes, tensor.dim());
        if (dims.empty()) {
            return {tensor, target};
        }

        auto selected_indices = Details::select_indices_by_frequency(tensor.size(0), opt.frequency, tensor.device());
        if (selected_indices.numel() == 0) {
            return {tensor, target};
        }

        auto selected_inputs = tensor.index_select(0, selected_indices).clone();
        auto target_indices = selected_indices.device() == target.device() ? selected_indices : selected_indices.to(target.device());
        auto selected_targets = target.index_select(0, target_indices).clone();

        auto flipped = selected_inputs.flip(dims);
        std::vector<int64_t> target_dims;
        target_dims.reserve(dims.size());
        const auto target_rank = selected_targets.dim();
        for (const auto dim : dims) {
            if (dim < target_rank && selected_targets.size(dim) == tensor.size(dim)) {
                target_dims.push_back(dim);
            }
        }
        auto flipped_targets = target_dims.empty() ? selected_targets : selected_targets.flip(target_dims);
        auto augmented_inputs = torch::cat({tensor, flipped}, 0);
        auto augmented_targets = torch::cat({target, flipped_targets}, 0);
        return {std::move(augmented_inputs), std::move(augmented_targets)};
    }
}

#endif // THOT_DATA_TRANSFORMS_AUGMENTATION_FLIP_HPP