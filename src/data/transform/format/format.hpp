#ifndef THOT_DATA_TRANSFORMS_FORMAT_HPP
#define THOT_DATA_TRANSFORMS_FORMAT_HPP

#include <array>
#include <optional>
#include <stdexcept>

#include <torch/torch.h>

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
    }

    inline torch::Tensor Upscale(const torch::Tensor& tensor, Options::UpscaleOptions options = {}) {
        [[maybe_unused]] const auto requested_size = options.targetsize;
        [[maybe_unused]] const bool show_progress = options.showprogress;
        if (!tensor.defined()) {
            throw std::invalid_argument("Format::Upscale expects a defined tensor.");
        }
        if (tensor.numel() == 0) {
            return Details::clone_as_dtype(tensor, torch::kUInt8);
        }

        auto float_tensor = Details::to_float32(tensor);
        auto scaled = float_tensor.mul(255.0f).clamp(0.0f, 255.0f).round();
        return scaled.to(tensor.options().dtype(torch::kUInt8));
    }

    inline torch::Tensor Downscale(const torch::Tensor& tensor, Options::DownscaleOptions options = {}) {
        [[maybe_unused]] const auto requested_size = options.targetsize;
        [[maybe_unused]] const bool show_progress = options.showprogress;
        if (!tensor.defined()) {
            throw std::invalid_argument("Format::Downscale expects a defined tensor.");
        }
        if (tensor.numel() == 0) {
            return Details::to_float32(tensor);
        }

        auto float_tensor = Details::to_float32(tensor);
        return float_tensor / 255.0f;
    }
}

namespace Thot::Data::Transform {
    namespace Format = ::Thot::Data::Transforms::Format;
}

#endif // THOT_DATA_TRANSFORMS_FORMAT_HPP