#ifndef THOT_RELIABILITY_PR_HPP
#define THOT_RELIABILITY_PR_HPP

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <torch/torch.h>

#include "../../../utils/terminal.hpp"
#include "reliability_curve_utils.hpp"
#include "../../../utils/gnuplot.hpp"
#include "../reliability.hpp"

namespace Thot {

    class Model;
}


namespace Thot::Plot::Details::Reliability {
    namespace detail {

        inline void RenderPRFromSeries(Model&, const Plot::Reliability::PRDescriptor& descriptor, std::vector<Curves::BinarySeries> series) {
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
            plotter.setGrid(true);
            plotter.setKey("top right");
            const auto& options = descriptor.options;
            if (options.logScale && options.expScale) {
                throw std::invalid_argument(
                    "Precision-Recall rendering cannot enable both logScale and expScale at the same time.");
            }

            constexpr double logEpsilon = 1e-6;

            if (options.logScale) {
                plotter.setLogScale('x');
                plotter.setLogScale('y');
                plotter.setRange('x', logEpsilon, 1.0);
                plotter.setRange('y', logEpsilon, 1.0);
            } else if (options.expScale) {
                const double expMax = std::expm1(1.0);
                plotter.setRange('x', 0.0, expMax);
                plotter.setRange('y', 0.0, expMax);
            } else {
                plotter.setRange('x', 0.0, 1.0);
                plotter.setRange('y', 0.0, 1.0);
            }


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
                    double recall = tp / static_cast<double>(curve.totalPositives);
                    const double precisionDenominator = tp + fp;
                    double precision = precisionDenominator > 0.0 ? tp / precisionDenominator : 1.0;

                    if (options.logScale) {
                        recall = recall <= 0.0 ? logEpsilon : recall;
                        precision = precision <= 0.0 ? logEpsilon : precision;
                    }

                    if (options.expScale) {
                        recall = std::expm1(recall);
                        precision = std::expm1(precision);
                    }
                    recallValues.push_back(recall);
                    precisionValues.push_back(precision);
                }

                if (recallValues.empty()) {
                    continue;
                }

                const double startThreshold = options.logScale ? logEpsilon : 0.0;
                if (recallValues.front() > startThreshold) {
                    double recall = 0.0;
                    double precision = 1.0;
                    if (options.logScale) {
                        recall = logEpsilon;
                        precision = 1.0;
                    }
                    if (options.expScale) {
                        recall = std::expm1(recall);
                        precision = std::expm1(precision);
                    }
                    recallValues.insert(recallValues.begin(), recall);
                    precisionValues.insert(precisionValues.begin(), precision);
                }
                if (recallValues.back() < 1.0) {
                    double recall = 1.0;
                    double precision = 0.0;
                    if (options.logScale) {
                        recall = 1.0;
                        precision = logEpsilon;
                    }
                    if (options.expScale) {
                        recall = std::expm1(recall);
                        precision = std::expm1(precision);
                    }
                    recallValues.push_back(recall);
                    precisionValues.push_back(precision);
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

    inline void RenderPR(Model& model, const Plot::Reliability::PRDescriptor& descriptor, torch::Tensor inputs, torch::Tensor targets)
    {
        std::vector<Curves::BinarySeries> series;
        series.emplace_back(Curves::MakeSeriesFromSamples(model, std::move(inputs), std::move(targets), ""));
        RenderPR(model, descriptor, std::move(series));
    }

    inline void RenderPR(Model& model, const Plot::Reliability::PRDescriptor& descriptor, torch::Tensor trainInputs, torch::Tensor trainTargets, torch::Tensor testInputs, torch::Tensor testTargets)
    {
        auto series = Curves::MakeSeriesFromSamples(model,
                                                    std::move(trainInputs),
                                                    std::move(trainTargets),
                                                    std::move(testInputs),
                                                    std::move(testTargets),
                                                    "Train",
                                                    "Test");
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