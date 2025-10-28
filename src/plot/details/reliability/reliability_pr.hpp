#ifndef THOT_RELIABILITY_PR_HPP
#define THOT_RELIABILITY_PR_HPP

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "reliability_curve_utils.hpp"
#include "../../../utils/gnuplot.hpp"

namespace Thot {
    class Model;
}

namespace Thot::Plot::Reliability {
    struct PRDescriptor;
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

        inline void RenderPRFromSeries(Model& /*model*/,
                                       const Plot::Reliability::PRDescriptor& descriptor,
                                       std::vector<Curves::BinarySeries> series)
        {
            if (series.empty()) {
                throw std::invalid_argument("Precision-Recall rendering requires at least one series.");
            }

            for (std::size_t index = 0; index < series.size(); ++index) {
                if (series[index].label.empty()) {
                    series[index].label = Curves::DefaultLabelForIndex(index);
                }
            }

            Utils::Gnuplot plotter{};
            plotter.setTitle("Precision-Recall Curve");
            plotter.setXLabel("Recall");
            plotter.setYLabel("Precision");
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

                std::vector<double> recallValues;
                std::vector<double> precisionValues;
                recallValues.reserve(curve.points.size());
                precisionValues.reserve(curve.points.size());

                for (const auto& point : curve.points) {
                    const double tp = static_cast<double>(point.truePositives);
                    const double fp = static_cast<double>(point.falsePositives);
                    const double recall = tp / static_cast<double>(curve.totalPositives);
                    const double precisionDenominator = tp + fp;
                    const double precision = precisionDenominator > 0.0 ? tp / precisionDenominator : 1.0;
                    recallValues.push_back(recall);
                    precisionValues.push_back(precision);
                }

                if (recallValues.empty()) {
                    continue;
                }

                if (recallValues.front() > 0.0) {
                    recallValues.insert(recallValues.begin(), 0.0);
                    precisionValues.insert(precisionValues.begin(), 1.0);
                }
                if (recallValues.back() < 1.0) {
                    recallValues.push_back(1.0);
                    precisionValues.push_back(0.0);
                }

                Utils::Gnuplot::PlotStyle style{};
                style.mode = Utils::Gnuplot::PlotMode::LinesPoints;
                style.lineWidth = 2.0;
                style.pointType = 7;
                style.pointSize = 1.1;
                style.lineColor = PickColor(index);

                datasets.push_back(Utils::Gnuplot::DataSet2D{
                    std::move(recallValues),
                    std::move(precisionValues),
                    entry.label,
                    style});
            }

            if (datasets.empty()) {
                throw std::runtime_error("Precision-Recall rendering could not assemble any datasets.");
            }

            plotter.plot(std::move(datasets));
        }
    } // namespace detail

    inline void RenderPR(Model& model,
                          const Plot::Reliability::PRDescriptor& descriptor,
                          std::vector<Curves::BinarySeries> series)
    {
        detail::RenderPRFromSeries(model, descriptor, std::move(series));
    }

    inline void RenderPR(Model& model,
                          const Plot::Reliability::PRDescriptor& descriptor,
                          torch::Tensor logits,
                          torch::Tensor targets)
    {
        std::vector<Curves::BinarySeries> series;
        series.emplace_back(Curves::MakeSeriesFromTensor(std::move(logits), std::move(targets), ""));
        RenderPR(model, descriptor, std::move(series));
    }

    inline void RenderPR(Model& model,
                          const Plot::Reliability::PRDescriptor& descriptor,
                          torch::Tensor trainLogits,
                          torch::Tensor trainTargets,
                          torch::Tensor testLogits,
                          torch::Tensor testTargets)
    {
        std::vector<Curves::BinarySeries> series;
        series.emplace_back(Curves::MakeSeriesFromTensor(std::move(trainLogits), std::move(trainTargets), "Train"));
        series.emplace_back(Curves::MakeSeriesFromTensor(std::move(testLogits), std::move(testTargets), "Test"));
        RenderPR(model, descriptor, std::move(series));
    }

    inline void RenderPR(Model& model,
                          const Plot::Reliability::PRDescriptor& descriptor,
                          const std::vector<double>& probabilities,
                          const std::vector<int64_t>& targets)
    {
        std::vector<Curves::BinarySeries> series;
        series.emplace_back(Curves::MakeSeriesFromVectors(probabilities, targets, ""));
        RenderPR(model, descriptor, std::move(series));
    }

    inline void RenderPR(Model& model,
                          const Plot::Reliability::PRDescriptor& descriptor,
                          const std::vector<double>& trainProbabilities,
                          const std::vector<int64_t>& trainTargets,
                          const std::vector<double>& testProbabilities,
                          const std::vector<int64_t>& testTargets)
    {
        std::vector<Curves::BinarySeries> series;
        series.emplace_back(Curves::MakeSeriesFromVectors(trainProbabilities, trainTargets, "Train"));
        series.emplace_back(Curves::MakeSeriesFromVectors(testProbabilities, testTargets, "Test"));
        RenderPR(model, descriptor, std::move(series));
    }
}


#endif //THOT_RELIABILITY_PR_HPP