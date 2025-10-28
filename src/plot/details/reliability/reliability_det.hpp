#ifndef THOT_RELIABILITY_DET_HPP
#define THOT_RELIABILITY_DET_HPP


#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "../../../utils/terminal.hpp"
#include "reliability_curve_utils.hpp"
#include "../../../utils/gnuplot.hpp"

namespace Thot {
    namespace Plot::Reliability {
        struct DETDescriptor;
    }

    class Model;
}


namespace Thot::Plot::Details::Reliability {
    namespace detail {
        inline auto PickColor(std::size_t index) -> std::string
        {
            static constexpr std::array<const char*, 8> palette{
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f"
            };
            return std::string(palette[index % palette.size()]);
        }

        inline void RenderDETFromSeries(Model& /*model*/,
                                        const Plot::Reliability::DETDescriptor& descriptor,
                                        std::vector<Curves::BinarySeries> series)
        {
            if (series.empty()) {
                throw std::invalid_argument("DET rendering requires at least one series.");
            }

            for (std::size_t index = 0; index < series.size(); ++index) {
                if (series[index].label.empty()) {
                    series[index].label = Curves::DefaultLabelForIndex(index);
                }
            }

            Utils::Gnuplot plotter{};
            plotter.setTitle("Detection Error Tradeoff");
            plotter.setXLabel("False Positive Rate");
            plotter.setYLabel("False Negative Rate");
            plotter.command("set xrange [0:1]");
            plotter.command("set yrange [0:1]");
            plotter.setGrid(true);
            plotter.setKey("top right");

            const auto& options = descriptor.options;
            (void)options;

            std::vector<Utils::Gnuplot::DataSet2D> datasets;
            datasets.reserve(series.size());

            for (std::size_t index = 0; index < series.size(); ++index) {
                const auto& entry = series[index];
                const auto curve = Curves::BuildCurve(entry);

                std::vector<double> falsePositiveRates;
                std::vector<double> falseNegativeRates;
                falsePositiveRates.reserve(curve.points.size());
                falseNegativeRates.reserve(curve.points.size());

                for (const auto& point : curve.points) {
                    const double fpr = static_cast<double>(point.falsePositives)
                        / static_cast<double>(curve.totalNegatives);
                    const double fnr = static_cast<double>(point.falseNegatives)
                        / static_cast<double>(curve.totalPositives);
                    falsePositiveRates.push_back(fpr);
                    falseNegativeRates.push_back(fnr);
                }

                Utils::Gnuplot::PlotStyle style{};
                style.mode = Utils::Gnuplot::PlotMode::LinesPoints;
                style.lineWidth = 2.0;
                style.pointType = 7;
                style.pointSize = 1.1;
                style.lineColor = PickColor(index);

                datasets.push_back(Utils::Gnuplot::DataSet2D{
                    std::move(falsePositiveRates),
                    std::move(falseNegativeRates),
                    entry.label,
                    style});
            }

            plotter.plot(std::move(datasets));
        }
    } // namespace detail

    inline void RenderDET(Model& model, const Plot::Reliability::DETDescriptor& descriptor, std::vector<Curves::BinarySeries> series) {
        detail::RenderDETFromSeries(model, descriptor, std::move(series));
    }

    inline void RenderDET(Model& model, const Plot::Reliability::DETDescriptor& descriptor, torch::Tensor logits, torch::Tensor targets) {
        std::vector<Curves::BinarySeries> series;
        series.emplace_back(Curves::MakeSeriesFromTensor(std::move(logits), std::move(targets), ""));
        RenderDET(model, descriptor, std::move(series));
    }

    inline void RenderDET(Model& model, const Plot::Reliability::DETDescriptor& descriptor, torch::Tensor trainLogits,
                        torch::Tensor trainTargets, torch::Tensor testLogits, torch::Tensor testTargets) {
        std::vector<Curves::BinarySeries> series;
        series.emplace_back(Curves::MakeSeriesFromTensor(std::move(trainLogits), std::move(trainTargets), "Train"));
        series.emplace_back(Curves::MakeSeriesFromTensor(std::move(testLogits), std::move(testTargets), "Test"));
        RenderDET(model, descriptor, std::move(series));
    }

    inline void RenderDET(Model& model, const Plot::Reliability::DETDescriptor& descriptor, const std::vector<double>& probabilities, const std::vector<int64_t>& targets) {
        std::vector<Curves::BinarySeries> series;
        series.emplace_back(Curves::MakeSeriesFromVectors(probabilities, targets, ""));
        RenderDET(model, descriptor, std::move(series));
    }

    inline void RenderDET(Model& model, const Plot::Reliability::DETDescriptor& descriptor, const std::vector<double>& trainProbabilities, const std::vector<int64_t>& trainTargets, const std::vector<double>& testProbabilities, const std::vector<int64_t>& testTargets) {
        std::vector<Curves::BinarySeries> series;
        series.emplace_back(Curves::MakeSeriesFromVectors(trainProbabilities, trainTargets, "Train"));
        series.emplace_back(Curves::MakeSeriesFromVectors(testProbabilities, testTargets, "Test"));
        RenderDET(model, descriptor, std::move(series));
    }
}

#endif //THOT_RELIABILITY_DET_HPP