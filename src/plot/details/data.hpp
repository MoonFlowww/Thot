#ifndef THOT_PLOT_DETAILS_DATA_HPP
#define THOT_PLOT_DETAILS_DATA_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace Thot::Plot::Data {
    namespace Details {
        inline auto as_cpu_contiguous(const torch::Tensor& tensor) -> torch::Tensor
        {
            if (!tensor.defined()) {
                throw std::invalid_argument("Plot::Data tensor must be defined");
            }
            auto result = tensor;
            if (result.device().is_cuda()) {
                result = result.cpu();
            }
            if (!result.is_contiguous()) {
                result = result.contiguous();
            }
            return result;
        }

        inline auto flatten_to_double_vector(const torch::Tensor& tensor) -> std::vector<double>
        {
            auto contiguous = as_cpu_contiguous(tensor);
            contiguous = contiguous.to(torch::kFloat32);
            std::vector<double> values;
            values.reserve(static_cast<std::size_t>(contiguous.numel()));
            auto flat = contiguous.view({contiguous.numel()});
            auto accessor = flat.accessor<float, 1>();

            for (int64_t i = 0; i < contiguous.numel(); ++i) {
                values.push_back(static_cast<double>(accessor[i]));
            }
            return values;
        }

        inline auto build_color_palette() -> std::vector<std::string>
        {
            static const std::array<const char*, 10> palette = {
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"};
            return {palette.begin(), palette.end()};
        }
    }

    class Timeserie {
    private:
        std::vector<double> m_xAxis{};
        std::vector<std::vector<double>> m_series{};
        std::vector<std::string> m_colors{};
        std::optional<double> m_testSeparator{};
    public:
        Timeserie(torch::Tensor values,
                  std::optional<std::size_t> testSeparator = std::nullopt)
        {
            initialize(std::move(values), testSeparator);
        }

        [[nodiscard]] auto xAxis() const noexcept -> const std::vector<double>&
        {
            return m_xAxis;
        }

        [[nodiscard]] auto series() const noexcept -> const std::vector<std::vector<double>>&
        {
            return m_series;
        }

        [[nodiscard]] auto colors() const noexcept -> const std::vector<std::string>&
        {
            return m_colors;
        }

        [[nodiscard]] auto hasMultipleInputs() const noexcept -> bool
        {
            return m_series.size() > 1;
        }

        [[nodiscard]] auto testSeparator() const noexcept -> const std::optional<double>&
        {
            return m_testSeparator;
        }

    private:
        void initialize(torch::Tensor values, std::optional<std::size_t> testSeparator)
        {
            if (!values.defined()) {
                throw std::invalid_argument("Timeserie values must be defined");
            }

            values = Details::as_cpu_contiguous(values);
            values = values.to(torch::kFloat32);
            if (values.dim() == 1) {
                const auto length = static_cast<std::size_t>(values.size(0));
                m_series.emplace_back();
                m_series.back().reserve(length);
                auto accessor = values.accessor<float, 1>();
                for (int64_t idx = 0; idx < values.size(0); ++idx) {
                    m_series.back().push_back(static_cast<double>(accessor[idx]));
                }
            } else if (values.dim() == 2) {
                const auto length = static_cast<std::size_t>(values.size(0));
                const auto features = static_cast<std::size_t>(values.size(1));
                auto accessor = values.accessor<float, 2>();
                m_series.resize(features);
                for (std::size_t feature = 0; feature < features; ++feature) {
                    auto& serie = m_series[feature];
                    serie.reserve(length);
                    for (int64_t time = 0; time < values.size(0); ++time) {
                        serie.push_back(static_cast<double>(accessor[time][static_cast<int64_t>(feature)]));
                    }
                }
            } else {
                throw std::invalid_argument("Timeserie expects a 1D or 2D tensor");
            }

            const auto length = m_series.empty() ? 0 : m_series.front().size();
            m_xAxis.resize(length);
            for (std::size_t index = 0; index < length; ++index) {
                m_xAxis[index] = static_cast<double>(index);
            }

            if (m_series.size() > 1U) {
                auto palette = Details::build_color_palette();
                m_colors.reserve(m_series.size());
                for (std::size_t idx = 0; idx < m_series.size(); ++idx) {
                    m_colors.push_back(palette[idx % palette.size()]);
                }
            } else {
                m_colors.emplace_back("#1f77b4");
            }

            if (testSeparator.has_value()) {
                const auto separator = testSeparator.value();
                if (separator >= length) {
                    throw std::out_of_range("Timeserie test separator outside of range");
                }
                m_testSeparator = static_cast<double>(separator);
            }
        }


    };

    class Image {
    private:
        std::vector<std::size_t> m_indices{};
        std::vector<torch::Tensor> m_selectedImages{};
        bool m_isColor{false};
    public:
        Image(torch::Tensor images, std::vector<std::size_t> indices) : m_indices(std::move(indices)) {
            initialize(std::move(images));
        }

        [[nodiscard]] auto selected() const noexcept -> const std::vector<torch::Tensor>& {
            return m_selectedImages;
        }

        [[nodiscard]] auto indices() const noexcept -> const std::vector<std::size_t>& {
            return m_indices;
        }

        [[nodiscard]] auto isColor() const noexcept -> bool {
            return m_isColor;
        }

    private:
        void initialize(torch::Tensor images)
        {
            if (!images.defined()) {
                throw std::invalid_argument("Image tensor must be defined");
            }
            images = Details::as_cpu_contiguous(images);

            if (images.dim() < 3 || images.dim() > 4) {
                throw std::invalid_argument("Image expects a 3D or 4D tensor");
            }

            const auto batchSize = static_cast<std::size_t>(images.size(0));
            for (auto index : m_indices) {
                if (index >= batchSize) {
                    throw std::out_of_range("Requested image index out of bounds");
                }
            }

            if (images.dim() == 4) {
                const auto channels = static_cast<std::size_t>(images.size(1));
                m_isColor = channels > 1;
                for (auto index : m_indices) {
                    auto slice = images[index].detach().clone();
                    m_selectedImages.push_back(std::move(slice));
                }
            } else { // dim == 3
                m_isColor = false;
                for (auto index : m_indices) {
                    auto slice = images[index].detach().clone();
                    m_selectedImages.push_back(std::move(slice));
                }
            }
        }


    };
}

#endif // THOT_PLOT_DETAILS_DATA_HPP