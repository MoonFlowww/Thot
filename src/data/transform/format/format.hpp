#ifndef THOT_DATA_TRANSFORM_FORMAT_HPP
#define THOT_DATA_TRANSFORM_FORMAT_HPP

#include <array>
#include <optional>
#include <stdexcept>
#include <vector>

#include <torch/torch.h>
#include <torch/nn/functional.h>

namespace Thot::Data::Transform::Format {
    namespace Options {
        struct ScaleOptions {
            std::optional<std::vector<int>> size{};
            bool showprogress = false;
        };

        using UpsampleOptions = ScaleOptions;
        using DownsampleOptions = ScaleOptions;
    }

    namespace Details {
        inline torch::Tensor to_float32(const torch::Tensor& tensor) {
            if (tensor.scalar_type() == torch::kFloat32) {
                return tensor;
            }
            return tensor.to(tensor.options().dtype(torch::kFloat32));
        }

        inline torch::Tensor clone_as_dtype(const torch::Tensor& tensor, torch::Dtype dtype) {
            return tensor.to(tensor.options().dtype(dtype));
        }
        inline torch::Tensor resize_spatial(const torch::Tensor& tensor, const std::vector<int>& target_size) {
            if (tensor.dim() < 3)
                throw std::invalid_argument("Format::resize_spatial expects a tensor with at least 3 dimensions (C, H, W).");
            if (target_size[0] <= 0 || target_size[1] <= 0)
                throw std::invalid_argument("Format::resize_spatial expects positive target dimensions.");
            if (target_size.size() != 2)
                throw std::invalid_argument("Format::resize_spatial expects exactly two target dimensions.");

            
            auto working = tensor;
            bool added_batch_dim = false;
            if (working.dim() == 3) {
                working = working.unsqueeze(0);
                added_batch_dim = true;
            } else if (working.dim() != 4) {
                throw std::invalid_argument("Format::resize_spatial currently supports tensors with 3 or 4 dimensions.");
            }

            auto options = torch::nn::functional::InterpolateFuncOptions()
                                .mode(torch::kBilinear)
                                .align_corners(false)
                                .size(std::vector<int64_t>{target_size[0], target_size[1]});

            auto resized = torch::nn::functional::interpolate(working, options);
            if (added_batch_dim) {
                resized = resized.squeeze(0);
            }

            return resized;
        }
    }

    inline torch::Tensor Upsample(const torch::Tensor& tensor, Options::UpsampleOptions options = {}) {
        [[maybe_unused]] const auto requested_size = options.size;
        [[maybe_unused]] const bool show_progress = options.showprogress;
        if (!tensor.defined()) {
            throw std::invalid_argument("Format::Upsample expects a defined tensor.");
        }
        if (tensor.numel() == 0) {
            return Details::clone_as_dtype(tensor, tensor.scalar_type());
        }

        auto float_tensor = Details::to_float32(tensor);
        if (requested_size.has_value())
            float_tensor = Details::resize_spatial(float_tensor, *requested_size);

        return Details::clone_as_dtype(float_tensor, tensor.scalar_type());
    }

    inline torch::Tensor Downsample(const torch::Tensor& tensor, Options::DownsampleOptions options = {}) {
        [[maybe_unused]] const auto requested_size = options.size;
        [[maybe_unused]] const bool show_progress = options.showprogress;
        if (!tensor.defined()) {
            throw std::invalid_argument("Format::Downsample expects a defined tensor.");
        }
        if (tensor.numel() == 0) {
            return Details::clone_as_dtype(tensor, tensor.scalar_type());
        }

        auto float_tensor = Details::to_float32(tensor);
        if (requested_size.has_value()) {
            float_tensor = Details::resize_spatial(float_tensor, *requested_size);
        }

        return Details::clone_as_dtype(float_tensor, tensor.scalar_type());
    }
}

#endif // THOT_DATA_TRANSFORM_FORMAT_HPP