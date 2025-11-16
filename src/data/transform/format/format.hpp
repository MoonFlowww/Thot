#ifndef THOT_DATA_TRANSFORMS_FORMAT_HPP
#define THOT_DATA_TRANSFORMS_FORMAT_HPP

#include <array>
#include <optional>
#include <stdexcept>
#include <vector>

#include <torch/torch.h>
#include <torch/nn/functional.h>

namespace Thot::Data::Transforms::Format {
    namespace Options {
        struct ScaleOptions {
            std::optional<std::array<int64_t, 2>> targetsize{};
            bool showprogress = false;
        };

        using UpscaleOptions = ScaleOptions;
        using DownscaleOptions = ScaleOptions;
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
        inline torch::Tensor resize_spatial(const torch::Tensor& tensor, const std::array<int64_t, 2>& target_size) {
            if (tensor.dim() < 3) {
                throw std::invalid_argument("Format::resize_spatial expects a tensor with at least 3 dimensions (C, H, W).");
            }

            if (target_size[0] <= 0 || target_size[1] <= 0) {
                throw std::invalid_argument("Format::resize_spatial expects positive target dimensions.");
            }

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

    inline torch::Tensor Upscale(const torch::Tensor& tensor, Options::UpscaleOptions options = {}) {
        [[maybe_unused]] const auto requested_size = options.targetsize;
        [[maybe_unused]] const bool show_progress = options.showprogress;
        if (!tensor.defined()) {
            throw std::invalid_argument("Format::Upscale expects a defined tensor.");
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

    inline torch::Tensor Downscale(const torch::Tensor& tensor, Options::DownscaleOptions options = {}) {
        [[maybe_unused]] const auto requested_size = options.targetsize;
        [[maybe_unused]] const bool show_progress = options.showprogress;
        if (!tensor.defined()) {
            throw std::invalid_argument("Format::Downscale expects a defined tensor.");
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

namespace Thot::Data::Transform {
    namespace Format = ::Thot::Data::Transforms::Format;
}

#endif // THOT_DATA_TRANSFORMS_FORMAT_HPP