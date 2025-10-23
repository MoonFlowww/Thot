#ifndef THOT_DIMREDUCTION_HPP
#define THOT_DIMREDUCTION_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <torch/torch.h>

#include "../../core.hpp"

namespace Thot::Data::DimReduction {
    namespace Details {
        inline torch::Tensor shrinkage(const torch::Tensor& tensor, double tau) {
            if (tau <= 0.0) {
                return tensor;
            }
            auto abs_tensor = tensor.abs();
            auto sign_tensor = torch::sign(tensor);
            auto clipped = torch::relu(abs_tensor - tau);
            return sign_tensor * clipped;
        }

        inline double safe_norm(const torch::Tensor& tensor, double default_value = 1.0) {
            if (tensor.numel() == 0) {
                return default_value;
            }
            auto value = tensor.norm().item<double>();
            if (!std::isfinite(value) || value <= 0.0) {
                return default_value;
            }
            return value;
        }

        inline double spectral_norm(const torch::Tensor& tensor) {
            auto svd = torch::linalg::svdvals(tensor);
            if (svd.numel() == 0) {
                return 0.0;
            }
            return svd.max().item<double>();
        }
    }

    template <bool BufferVRAM = false, class DevicePolicyT = Core::DevicePolicy<BufferVRAM>>
    [[nodiscard]] inline std::pair<torch::Tensor, torch::Tensor>
    RPCA(const torch::Tensor& input,
         double lambda = -1.0,
         double tolerance = 1e-7,
         std::size_t max_iterations = 1000,
         double rho = 1.5,
         bool keep_on_device = BufferVRAM) {
        if (input.dim() != 2) {
            throw std::invalid_argument("RPCA expects a two-dimensional tensor (observations x features).");
        }

        auto device = torch::Device(torch::kCPU);
        if constexpr (BufferVRAM) {
            device = DevicePolicyT::select();
        }

        auto data = input.to(device, torch::kFloat32).clone();
        if (!data.is_contiguous()) {
            data = data.contiguous();
        }

        const auto rows = data.size(0);
        const auto cols = data.size(1);

        if (lambda <= 0.0) {
            lambda = 1.0 / std::sqrt(static_cast<double>(std::max(rows, cols)));
        }

        const auto norm_two = Details::spectral_norm(data);
        const auto norm_inf = data.abs().sum(1).amax().item<double>();
        const auto dual_norm = std::max(norm_two, norm_inf / lambda);

        auto Y = data / (dual_norm > 0.0 ? dual_norm : 1.0);
        double mu = 1.25 / (norm_two > 0.0 ? norm_two : 1.0);
        const double mu_bar = mu * 1e7;

        auto L = torch::zeros_like(data);
        auto S = torch::zeros_like(data);

        const double fro_norm = Details::safe_norm(data, 1.0);

        std::size_t iteration = 0;
        double error = std::numeric_limits<double>::max();

        while (iteration < max_iterations && error > tolerance) {
            const auto current = data - S + (1.0 / mu) * Y;
            auto svd = torch::linalg::svd(current, /*full_matrices=*/false);
            auto U = std::get<0>(svd);
            auto Sigma = std::get<1>(svd);
            auto Vh = std::get<2>(svd);

            auto sigma_threshold = Details::shrinkage(Sigma, 1.0 / mu);
            auto diag_sigma = torch::diag(sigma_threshold);
            auto L_next = U.matmul(diag_sigma).matmul(Vh);

            auto temp = data - L_next + (1.0 / mu) * Y;
            auto S_next = Details::shrinkage(temp, lambda / mu);

            auto residual = data - L_next - S_next;
            Y = Y + mu * residual;
            mu = std::min(mu * rho, mu_bar);

            error = residual.norm().item<double>() / fro_norm;
            L = L_next;
            S = S_next;
            ++iteration;
        }

        if (!keep_on_device) {
            L = L.to(torch::kCPU);
            S = S.to(torch::kCPU);
        }

        return {L, S};
    }

    struct PCAResult {
        torch::Tensor components;
        torch::Tensor explained_variance;
        torch::Tensor singular_values;
        torch::Tensor mean;
        torch::Tensor transformed;
    };

    template <bool BufferVRAM = false, class DevicePolicyT = Core::DevicePolicy<BufferVRAM>>
    [[nodiscard]] inline PCAResult PCA(const torch::Tensor& input,
                                       std::size_t components = 0,
                                       bool center = true,
                                       bool whiten = false,
                                       bool keep_on_device = BufferVRAM) {
        if (input.dim() != 2) {
            throw std::invalid_argument("PCA expects a two-dimensional tensor (observations x features).");
        }

        auto device = torch::Device(torch::kCPU);
        if constexpr (BufferVRAM) {
            device = DevicePolicyT::select();
        }

        auto data = input.to(device, torch::kFloat32);
        if (!data.is_contiguous()) {
            data = data.contiguous();
        }

        const auto samples = data.size(0);
        const auto features = data.size(1);

        if (components == 0 || components > static_cast<std::size_t>(std::min(samples, features))) {
            components = static_cast<std::size_t>(std::min(samples, features));
        }

        torch::Tensor mean = torch::zeros({features}, data.options());
        if (center) {
            mean = data.mean(0);
            data = data - mean;
        }

        auto svd = torch::linalg_svd(data, /*full_matrices=*/false);
        auto U = std::get<0>(svd);
        auto S = std::get<1>(svd);
        auto Vh = std::get<2>(svd);


        const auto component_count = static_cast<int64_t>(components);
        auto selected_singular = S.narrow(0, 0, component_count);
        auto components_matrix = Vh.narrow(0, 0, component_count);

        auto explained_variance = (selected_singular.pow(2)) / std::max<int64_t>(samples - 1, 1);
        auto transformed = U.narrow(1, 0, component_count) * selected_singular;

        if (whiten) {
            auto eps = 1e-12f;
            auto scaling = torch::reciprocal(torch::sqrt(explained_variance + eps));
            transformed = transformed * scaling;
        }

        if (!keep_on_device) {
            components_matrix = components_matrix.to(torch::kCPU);
            explained_variance = explained_variance.to(torch::kCPU);
            selected_singular = selected_singular.to(torch::kCPU);
            mean = mean.to(torch::kCPU);
            transformed = transformed.to(torch::kCPU);
        }

        return {components_matrix, explained_variance, selected_singular, mean, transformed};
    }

    template <bool BufferVRAM = false, class DevicePolicyT = Core::DevicePolicy<BufferVRAM>>
    [[nodiscard]] inline torch::Tensor ProjectPCA(const torch::Tensor& input,
                                                 const PCAResult& pca,
                                                 bool keep_on_device = BufferVRAM) {
        auto device = torch::Device(torch::kCPU);
        if constexpr (BufferVRAM) {
            device = DevicePolicyT::select();
        }

        auto components = keep_on_device ? pca.components.to(device) : pca.components.to(torch::kCPU);
        auto mean = keep_on_device ? pca.mean.to(device) : pca.mean.to(torch::kCPU);
        auto data = input.to(components.device(), torch::kFloat32);
        if (mean.numel() == data.size(1)) {
            data = data - mean;
        }

        auto projected = torch::matmul(data, components.transpose(0, 1));

        if (!keep_on_device && projected.device().is_cuda()) {
            projected = projected.to(torch::kCPU);
        }
        return projected;
    }
}

#endif //THOT_DIMREDUCTION_HPP